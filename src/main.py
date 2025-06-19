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

from model import ConvNet
import util

@dataclass
class ModelArgs:
    layers: int = 8
    channels: int = 64
    out_channels: int = 2
    n_dts: int = 1

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

    model: ModelArgs = ModelArgs()
    opt: OptimizerArgs = OptimizerArgs()
    data: DataArgs = DataArgs()

class Main:
    def __init__(self, args: Args):
        if isinstance(args.data.dt, int):
            args.data.dt = [args.data.dt]
        print(args)
        self.dts = jnp.array(args.data.dt)
        self.dt_max = self.dts.max().item()

        self.net = ConvNet(layers=args.model.layers, channels=args.model.channels, out_channels=args.model.out_channels, n_dts=args.model.n_dts)
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
        x0, x1, dt_id = batch['x0'], batch['x1'], batch['dt_id']
        logits = jax.vmap(self.net.apply, in_axes=(None, 0, 0))(params, x0, dt_id) # (B, H, W, 2)
        labels = jax.nn.one_hot(x1, 2) # (B, H, W, 2)

        loss_ce = jax.vmap(jax.vmap(jax.vmap(optax.softmax_cross_entropy)))(logits, labels)
        loss = loss_ce.mean()
        metrics = dict(loss=loss, loss_ce=loss_ce, logits=logits, labels=labels)
        return loss, metrics

    def do_iter(self, train_state, rng, train=True):
        batch = self.generate_batch(rng)
        (_, metrics), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(train_state.params, batch)
        if train:
            train_state = train_state.apply_gradients(grads=grads)
        return train_state, metrics
    
    def run(self):
        rng = jax.random.PRNGKey(self.args.seed)
        instance = self.generate_instance(rng)
        params = self.net.init(rng, instance['x0'], instance['dt_id'])
        n_params = sum(x.size for x in jax.tree.leaves(params))
        print(f"Number of parameters: {n_params:,}")

        tx = optax.chain(optax.clip_by_global_norm(self.args.opt.clip_grad_norm),
                         optax.adamw(self.args.opt.learning_rate, weight_decay=self.args.opt.weight_decay, eps=1e-8))
        train_state = TrainState.create(apply_fn=self.net.apply, params=params, tx=tx)

        final_loss = np.inf
        loss_history = []
        pbar = tqdm(range(self.args.n_iters))
        for i_iter in pbar:
            rng, _rng = split(rng)
            train_state, metrics = self.do_iter_train(train_state, _rng)

            loss_history.append(metrics['loss'].item())
            if self.args.save_dir is not None and (i_iter % self.args.log_every == 0 or i_iter == self.args.n_iters - 1):
                os.makedirs(self.args.save_dir, exist_ok=True)
                util.save_pkl(self.args.save_dir, "args", self.args)
                util.save_pkl(self.args.save_dir, "loss_history", loss_history)
                _, metrics = jax.lax.scan(self.do_iter_eval, train_state, split(rng, 100))
                final_loss_now = metrics['loss'].mean().item()
                if final_loss_now < final_loss:
                    final_loss = final_loss_now
                    util.save_pkl(self.args.save_dir, "final_loss", final_loss)
                    util.save_pkl(self.args.save_dir, "params", jax.tree.map(lambda x: np.array(x), train_state.params))
            pbar.set_postfix(loss=loss_history[-1], ppl=np.exp(loss_history[-1]))


if __name__ == "__main__":
    Main(tyro.cli(Args)).run()

