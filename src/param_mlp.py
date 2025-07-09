import os
from functools import partial
from dataclasses import dataclass, field
from typing import List, Optional
from tqdm import tqdm
import tyro

import jax
import jax.numpy as jnp
from jax.random import split, PRNGKey
import optax
from flax.training.train_state import TrainState
import flax.linen as nn

from substrates.gol import GameOfLife
from rollout import rollout_simulation
from foundation_models import create_foundation_model
import util

import numpy as np


@dataclass
class OptimizerArgs:
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    clip_grad_norm: float = 1.0

@dataclass
class DataArgs:
    gol_params_max: int = 262144
    grid_size: int = 64
    t_start: int = 32
    t_end: int = 64
    dt: int | List[int] = 1

@dataclass
class MLPModelArgs:
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    output_dim: int = 512

@dataclass
class ParamMLPArgs:
    seed: int = 0
    save_dir: Optional[str] = None
    n_iters: int = 1000
    log_every: int = 100

    model: MLPModelArgs = field(default_factory=MLPModelArgs)
    opt: OptimizerArgs = field(default_factory=OptimizerArgs)
    data: DataArgs = field(default_factory=DataArgs)

class ParamMLP(nn.Module):
    hidden_dims: List[int]
    output_dim: int

    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        return nn.Dense(self.output_dim)(x)


def main(args: ParamMLPArgs):
    # Use only open-ended rules
    params_np = np.load("low_oe_params.npy")
    params_jnp = jnp.array(params_np, dtype=jnp.int32)
    n_saved = params_jnp.shape[0]

    dts = jnp.array([args.data.dt] if isinstance(args.data.dt, int) else args.data.dt)
    dt_max = int(dts.max())

    substrate = GameOfLife(grid_size=args.data.grid_size)
    fm = create_foundation_model('clip')
    mlp = ParamMLP(hidden_dims=args.model.hidden_dims, output_dim=args.model.output_dim)
    
    rollout_fn = jax.jit(partial(
        rollout_simulation,
        s0=None, substrate=substrate, fm=None,
        rollout_steps=args.data.t_end + dt_max,
        time_sampling='video', img_size=None,
        return_state=True
    ))
    num_bits = int(jnp.ceil(jnp.log2(args.data.gol_params_max)))

    # Generate a single training sample
    def gen_one(rng):
        rng, sub = split(rng)
        # pid = jax.random.randint(sub, (), 0, args.data.gol_params_max, dtype=jnp.int32)
        idx = jax.random.randint(sub, (), 0, n_saved, dtype=jnp.int32)  # Only use open-ended params
        pid = params_jnp[idx]
        rng, sub = split(rng)
        state = rollout_fn(sub, pid)['state']
        rng, sub = split(rng)
        t0 = jax.random.randint(sub, (), args.data.t_start, args.data.t_end, dtype=jnp.int32)
        rng, sub = split(rng)
        dt_id = jax.random.randint(sub, (), 0, len(dts), dtype=jnp.int32)
        x1 = state[t0 + dts[dt_id]]
        bits = ((pid[..., None] >> jnp.arange(num_bits)) & 1).astype(jnp.float32)
        return bits, x1, pid

    # Generate a batch of training samples
    generate_batch = jax.jit(jax.vmap(gen_one))

    @jax.jit
    def train_step(state, rng):
        # Sample a batch
        rngs = jax.random.split(rng, args.opt.batch_size)
        bits, states, pids = generate_batch(rngs)
        render_fn = jax.vmap(lambda state, pid: substrate.render_state(state, params=pid, img_size=224))
        imgs = render_fn(states, pids)
        embeds = jax.vmap(fm.embed_img)(imgs)
        def loss_fn(params):
            preds = mlp.apply({'params': params}, bits)
            return jnp.mean((preds - embeds)**2)
        grads = jax.grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        loss = loss_fn(state.params)
        return state, loss

    # Initialization
    rng = PRNGKey(args.seed)
    rng, init_rng = split(rng)
    params0 = mlp.init(init_rng, jnp.zeros((args.opt.batch_size, num_bits)))['params']
    tx = optax.chain(
        optax.clip_by_global_norm(args.opt.clip_grad_norm),
        optax.adamw(args.opt.learning_rate, weight_decay=args.opt.weight_decay)
    )
    state = TrainState.create(apply_fn=mlp.apply, params=params0, tx=tx)

    # Training loop
    history = []
    rng = rng
    for i in tqdm(range(args.n_iters)):
        rng, step_rng = split(rng)
        state, loss = train_step(state, step_rng)
        history.append(loss)
        if i % args.log_every == 0:
            tqdm.write(f"Iter {i}: loss={loss:.4f}")

    # Save results
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        util.save_pkl(args.save_dir, 'params', state.params)
        util.save_pkl(args.save_dir, 'history', history)


if __name__ == '__main__':
    main(tyro.cli(ParamMLPArgs))
