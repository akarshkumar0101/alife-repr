import os
from functools import partial
from dataclasses import dataclass, field
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
    n_iters: int = 1000
    log_every: int = 100

    model: ModelArgs = field(default_factory=ModelArgs)
    opt: OptimizerArgs = field(default_factory=OptimizerArgs)
    data: DataArgs = field(default_factory=DataArgs)

def create_net(args: Args):
    return ConvNet(layers=args.model.layers, channels=args.model.channels, out_channels=args.model.out_channels, n_dts=args.model.n_dts)

def main(args: Args):
    if isinstance(args.data.dt, int):
        args.data.dt = [args.data.dt]
    print(args)
    dts = jnp.array(args.data.dt)
    dt_max = dts.max().item()

    net = create_net(args)
    substrate = GameOfLife(grid_size=args.data.grid_size)
    rollout_fn = partial(rollout_simulation, s0=None, substrate=substrate, fm=None, rollout_steps=args.data.t_end+dt_max,
                         time_sampling='video', img_size=None, return_state=True)
    
    def generate_batch(rng):
        rng, _rng = split(rng)
        state = rollout_fn(_rng, args.data.gol_params)['state']

        rng, _rng = split(rng)
        t0 = jax.random.randint(_rng, shape=(), minval=args.data.t_start, maxval=args.data.t_end)

        rng, _rng = split(rng)
        dt_id = jax.random.randint(_rng, shape=(), minval=0, maxval=len(dts))
        dt = dts[dt_id]
        t1 = t0 + dt
        x0, x1 = state[t0], state[t1]
        return dict(x0=x0, x1=x1, dt_id=dt_id, dt=dt, state=state, t0=t0, t1=t1)

    def loss_fn(params, batch):
        x0, x1, dt_id = batch['x0'], batch['x1'], batch['dt_id']
        logits = jax.vmap(net.apply, in_axes=(None, 0, 0))(params, x0, dt_id) # (B, H, W, 2)
        labels = jax.nn.one_hot(x1, 2) # (B, H, W, 2)

        loss_ce = jax.vmap(jax.vmap(jax.vmap(optax.softmax_cross_entropy)))(logits, labels)
        loss = loss_ce.mean()
        metrics = dict(loss=loss, loss_ce=loss_ce)
        return loss, metrics

    @jax.jit
    def iter_train(train_state, batch):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params, batch)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, metrics

    rng = jax.random.PRNGKey(args.seed)
    batch = generate_batch(rng)
    params = net.init(rng, batch['x0'], batch['dt_id'])
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"Number of parameters: {n_params:,}")
    generate_batch_vmap = jax.jit(jax.vmap(generate_batch))

    tx = optax.chain(optax.clip_by_global_norm(args.opt.clip_grad_norm), optax.adamw(args.opt.learning_rate, weight_decay=args.opt.weight_decay, eps=1e-8))
    train_state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    loss_history = []
    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        rng, _rng = split(rng)
        batch = generate_batch_vmap(split(_rng, args.opt.batch_size))

        train_state, metrics = iter_train(train_state, batch)
        loss_history.append(metrics['loss'].item())
        if args.save_dir is not None and (i_iter % args.log_every == 0 or i_iter == args.n_iters - 1):
            os.makedirs(args.save_dir, exist_ok=True)
            util.save_pkl(args.save_dir, "params", jax.tree.map(lambda x: np.array(x), train_state.params))
            util.save_pkl(args.save_dir, "args", args)
            util.save_pkl(args.save_dir, "loss_history", loss_history)
        pbar.set_postfix(loss=loss_history[-1], ppl=np.exp(loss_history[-1]))

if __name__ == "__main__":
    main(tyro.cli(Args))


