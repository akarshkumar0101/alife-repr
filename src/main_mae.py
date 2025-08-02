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

from models.mae import MAE, MAEConfig
import util

@dataclass
class OptimizerArgs:
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
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

    model: MAEConfig = MAEConfig(dt=1, nt=4, nw=16, nh=16, pt=1, ph=4, pw=4)
    opt: OptimizerArgs = OptimizerArgs()
    data: DataArgs = DataArgs()

class Main:
    def __init__(self, args: Args):
        print(args)
        self.sim_steps = args.model.dt * args.model.nt * args.model.pt

        self.net = MAE(args.model)
        self.substrate = GameOfLife(grid_size=args.data.grid_size)
        self.rollout_fn = partial(rollout_simulation, s0=None, substrate=self.substrate, fm=None, rollout_steps=args.data.t_end+self.sim_steps,
                                  time_sampling='video', img_size=None, return_state=True)
        self.args = args

        self.generate_batch = jax.jit(self.generate_batch)
        self.do_iter_train = jax.jit(partial(self.do_iter, train=True))
        self.do_iter_eval = jax.jit(partial(self.do_iter, train=False))
    
    def generate_instance(self, rng):
        _rng1, _rng2 = split(rng)
        state = self.rollout_fn(_rng1, self.args.data.gol_params)['state']
        t0 = jax.random.randint(_rng2, shape=(), minval=self.args.data.t_start, maxval=self.args.data.t_end)
        t1 = t0 + self.sim_steps
        # x = state[t0:t1:self.args.model.dt]
        x = jax.lax.dynamic_slice(state, (t0, 0, 0), (self.sim_steps, self.args.data.grid_size, self.args.data.grid_size))[::self.args.model.dt]
        return dict(rng=rng, x=x, t0=t0, t1=t1)

    def generate_batch(self, rng):
        return jax.vmap(self.generate_instance)(split(rng, self.args.opt.batch_size))

    def loss_fn(self, params, batch):
        rng, x = batch['rng'], batch['x']
        loss, metrics = jax.vmap(self.net.apply, in_axes=(None, 0, 0))(params, rng, x)
        # loss = loss.mean()
        loss = metrics['loss_masked'].mean()
        return loss, metrics

    def do_iter(self, train_state, rng, train=True):
        batch = self.generate_batch(rng)
        (_, metrics), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(train_state.params, batch)
        if train:
            train_state = train_state.apply_gradients(grads=grads)
        return train_state, metrics

    def init(self):
        rng = jax.random.PRNGKey(self.args.seed)
        instance = self.generate_instance(rng)
        params = self.net.init(rng, instance['rng'], instance['x'])
        n_params = sum(x.size for x in jax.tree.leaves(params))
        print(f"Number of parameters: {n_params:,}")

        tx = optax.chain(optax.clip_by_global_norm(self.args.opt.clip_grad_norm),
                         optax.adamw(self.args.opt.learning_rate, weight_decay=self.args.opt.weight_decay, eps=1e-8))
        self.train_state = TrainState.create(apply_fn=self.net.apply, params=params, tx=tx)

        final_loss = np.inf
        self.loss_history = []
    
    def run(self):
        rng = jax.random.PRNGKey(self.args.seed)
        pbar = tqdm(range(self.args.n_iters))
        for i_iter in pbar:
            rng, _rng = split(rng)
            self.train_state, metrics = self.do_iter_train(self.train_state, _rng)

            self.loss_history.append(metrics['loss_masked'].mean().item())
            # if self.args.save_dir is not None and (i_iter % self.args.log_every == 0 or i_iter == self.args.n_iters - 1):
            #     os.makedirs(self.args.save_dir, exist_ok=True)
            #     util.save_pkl(self.args.save_dir, "args", self.args)
            #     util.save_pkl(self.args.save_dir, "loss_history", loss_history)
            #     _, metrics = jax.lax.scan(self.do_iter_eval, train_state, split(rng, 100))
            #     final_loss_now = metrics['loss'].mean().item()
            #     if final_loss_now < final_loss:
            #         final_loss = final_loss_now
            #         util.save_pkl(self.args.save_dir, "final_loss", final_loss)
            #         util.save_pkl(self.args.save_dir, "params", jax.tree.map(lambda x: np.array(x), train_state.params))
            pbar.set_postfix(loss=self.loss_history[-1], ppl=np.exp(self.loss_history[-1]))


if __name__ == "__main__":
    main = Main(tyro.cli(Args))
    main.init()
    main.run()

