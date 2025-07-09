import os
from functools import partial
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
import tyro

import numpy as np
import jax
import jax.numpy as jnp
from jax.random import split
import optax
from flax.training.train_state import TrainState

from substrates.gol import GameOfLife
from rollout import rollout_simulation

from models.model_dt import DTNetwork, DTNetworkConfig
import util

@dataclass
class OptimizerArgs:
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.
    clip_grad_norm: float = 1.0

@dataclass
class DataArgs:
    gol_params: int = 6152
    grid_size: int = 64
    t_start: int = 32
    t_end: int = 64
    dt: int | List[int] = 1

@dataclass
class Args:
    seed: int = 0
    save_dir: str | None = None
    n_iters: int = 10000
    log_every: int = 1000

    model: DTNetworkConfig = DTNetworkConfig()
    opt: OptimizerArgs = OptimizerArgs()
    data: DataArgs = DataArgs()

class Main:
    def __init__(self, args: Args):
        if isinstance(args.data.dt, int):
            args.data.dt = [args.data.dt]
        print(args)
        self.dts = jnp.array(args.data.dt)
        self.dt_max = self.dts.max().item()

        self.net = DTNetwork(args.model)
        self.substrate = GameOfLife(grid_size=args.data.grid_size)
        self.rollout_fn = partial(rollout_simulation, s0=None, substrate=self.substrate, fm=None, rollout_steps=args.data.t_end+self.dt_max,
                                  time_sampling='video', img_size=None, return_state=True)
        self.args = args

        self.generate_batch = jax.jit(self.generate_batch)
        self.do_iter_train = jax.jit(partial(self.do_iter, train=True))
        self.do_iter_eval = jax.jit(partial(self.do_iter, train=False))
    
    def generate_instance(self, rng):
        rng, _rng = split(rng)
        state = self.rollout_fn(_rng, self.args.data.gol_params)['state']

        rng, _rng = split(rng)
        t0 = jax.random.randint(_rng, shape=(), minval=self.args.data.t_start, maxval=self.args.data.t_end)

        rng, _rng = split(rng)
        dt_id = jax.random.randint(_rng, shape=(), minval=0, maxval=len(self.dts))
        dt = self.dts[dt_id]
        t1 = t0 + dt
        x0, x1 = state[t0], state[t1]
        return dict(x0=x0, x1=x1, dt_id=dt_id, dt=dt, state=state, t0=t0, t1=t1)

    def generate_batch(self, rng):
        return jax.vmap(self.generate_instance)(split(rng, self.args.opt.batch_size))

    def loss_fn(self, params, batch):
        x0, x1 = batch['x0'], batch['x1']
        loss, metrics = jax.vmap(self.net.apply, in_axes=(None, 0, 0))(params, x0, x1)
        return loss.mean(), metrics

    def do_iter(self, train_state, rng, train=True):
        batch = self.generate_batch(rng)
        (_, metrics), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(train_state.params, batch)
        if train:
            train_state = train_state.apply_gradients(grads=grads)
        return train_state, {'grads': grads, **metrics}
    
    def init(self):
        rng = jax.random.PRNGKey(self.args.seed)
        instance = self.generate_instance(rng)
        self.init_params = self.net.init(rng, instance['x0'], instance['x1'])
        n_params = sum(x.size for x in jax.tree.leaves(self.init_params))
        print(f"Number of parameters: {n_params:,}")

        tx = optax.chain(optax.clip_by_global_norm(self.args.opt.clip_grad_norm),
                         optax.adamw(self.args.opt.learning_rate, weight_decay=self.args.opt.weight_decay, eps=1e-8))
        self.train_state = TrainState.create(apply_fn=self.net.apply, params=self.init_params, tx=tx)

        self.final_loss = np.inf
        self.loss_history = []

        self.rng = jax.random.PRNGKey(self.args.seed)
        self.i_iter = 0
    
    def step(self):
        self.rng, _rng = split(self.rng)
        self.train_state, metrics = self.do_iter_train(self.train_state, _rng)

        self.loss_history.append(metrics['loss'].mean().item())
        if self.args.save_dir is not None and (self.i_iter % self.args.log_every == 0 or self.i_iter == self.args.n_iters - 1):
            os.makedirs(self.args.save_dir, exist_ok=True)
            util.save_pkl(self.args.save_dir, "args", self.args)
            util.save_pkl(self.args.save_dir, "loss_history", self.loss_history)
            _, metrics = jax.lax.scan(self.do_iter_eval, self.train_state, split(_rng, 100))
            final_loss_now = metrics['loss'].mean().item()
            if final_loss_now < self.final_loss:
                self.final_loss = final_loss_now
                util.save_pkl(self.args.save_dir, "final_loss", self.final_loss)
                util.save_pkl(self.args.save_dir, "params", jax.tree.map(lambda x: np.array(x), self.train_state.params))
        
        self.i_iter += 1
    
    def run(self):
        pbar = tqdm(range(self.args.n_iters))
        for _ in pbar:
            self.step()
            pbar.set_postfix(loss=self.loss_history[-1], ppl=np.exp(self.loss_history[-1]))


if __name__ == "__main__":
    main = Main(tyro.cli(Args))
    main.init()
    main.run()

