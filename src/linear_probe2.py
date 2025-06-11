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
import flax.linen as nn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from substrates.gol import GameOfLife
from rollout import rollout_simulation
from foundation_models import create_foundation_model

from model import ConvNet
import util

from main import *

@dataclass
class LinearProbeArgs:
    seed: int = 0
    load_dir: str | None = None
    save_dir: str | None = None

    n_iters: int = 1000
    log_every: int = 100
    target: str = 'x0'  # 'x0' or 'x1'

    opt: OptimizerArgs = field(default_factory=OptimizerArgs)
    zero_features: bool | None = False # use zero features (for getting baseline)

class LinearProbe(nn.Module):
    d_in: int
    d_out: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.d_out, use_bias=True, kernel_init=nn.initializers.normal(0.01))(x)
        return x

def main(lp_args: LinearProbeArgs):
    args = util.load_pkl(lp_args.load_dir, "args")
    params = util.load_pkl(lp_args.load_dir, "params")
    fm = create_foundation_model('clip')

    # ----- COPIED FROM MAIN.PY -----
    if isinstance(args.data.dt, int):
        args.data.dt = [args.data.dt]
    print(lp_args)
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
    # ------------------------------

    def get_features(params, batch):
        x0, x1, dt_id = batch['x0'], batch['x1'], batch['dt_id']
        forward_fn = jax.vmap(partial(net.apply, return_hidden_reprs=True), in_axes=(None, 0, 0))
        y, hidden_reprs = forward_fn(params, x0, dt_id)
        features = jax.tree.map(lambda x: x.mean(axis=(-3, -2)), hidden_reprs)
        features = jnp.concatenate(features, axis=-1)
        return features

    def get_target(batch):
        if lp_args.target == 'x0':
            target = batch['x0']
        elif lp_args.target == 'x1':
            target = batch['x1']
        else:
            raise ValueError(f"Unknown target: {lp_args.target}")
        render_fn = jax.vmap(partial(substrate.render_state, params=args.data.gol_params, img_size=224))
        img = render_fn(target)
        z = jax.vmap(fm.embed_img)(img)
        return z
    
    linear_probe = LinearProbe(d_in=args.model.channels*args.model.layers, d_out=512)
        
    def loss_fn(lp_params, batch):
        features = get_features(params, batch)
        if lp_args.zero_features:
            features = jnp.zeros_like(features)
        y = get_target(batch)
        y_pred = jax.vmap(linear_probe.apply, in_axes=(None, 0))(lp_params, features)
        loss_mse = (y - y_pred)**2
        loss = loss_mse.mean()
        metrics = dict(loss=loss, loss_mse=loss_mse)
        # y_pred = y_pred / (jnp.linalg.norm(y_pred, axis=-1, keepdims=True) + 1e-8)
        # loss_cosine = -(y*y_pred).sum(axis=-1)
        # loss = loss_cosine.mean()
        # metrics = dict(loss=loss, loss_cosine=loss_cosine)
        return loss, metrics

    @jax.jit
    def iter_train(train_state, batch):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params, batch)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, metrics

    rng = jax.random.PRNGKey(lp_args.seed)

    generate_batch_vmap = jax.jit(jax.vmap(generate_batch))

    features = get_features(params, generate_batch_vmap(split(rng, 1)))[0]
    lp_params = linear_probe.init(rng, features)

    tx = optax.chain(optax.clip_by_global_norm(lp_args.opt.clip_grad_norm), optax.adamw(lp_args.opt.learning_rate, weight_decay=lp_args.opt.weight_decay, eps=1e-8))
    train_state = TrainState.create(apply_fn=linear_probe.apply, params=lp_params, tx=tx)

    rng = jax.random.PRNGKey(lp_args.seed)
    X, Y = [], []
    for i_iter in tqdm(range(500)):
        rng, _rng = split(rng)
        batch = generate_batch_vmap(split(_rng, lp_args.opt.batch_size))
        X.append(get_features(params, batch))
        Y.append(get_target(batch))
    X = jnp.concatenate(X, axis=0)
    Y = jnp.concatenate(Y, axis=0)

    if lp_args.zero_features:
        X = jnp.zeros_like(X)
    X, Y = np.array(X), np.array(Y)

    # reg = LinearRegression(fit_intercept=True).fit(X, Y)
    # score = reg.score(X, Y)
    # Y_pred = reg.predict(X)
    # mse = ((Y-Y_pred)**2).mean()
    # Y_pred_norm = Y_pred / (jnp.linalg.norm(Y_pred, axis=-1, keepdims=True) + 1e-8)
    # cossim = (Y*Y_pred_norm).sum(axis=-1).mean()
    # print(score, mse, cossim)

    # split into train & validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=lp_args.seed)

    # fit on training set
    reg = LinearRegression(fit_intercept=True).fit(X_train, Y_train)
    # evaluate on validation set
    val_score = reg.score(X_val, Y_val)
    Y_val_pred = reg.predict(X_val)
    val_mse = ((Y_val - Y_val_pred) ** 2).mean()
    # cosine similarity on validation
    Y_val_pred_norm = Y_val_pred / (jnp.linalg.norm(Y_val_pred, axis=-1, keepdims=True) + 1e-8)
    val_cossim = (Y_val * Y_val_pred_norm).sum(axis=-1).mean()
    print(f"Validation R²: {val_score:.4f}, MSE: {val_mse:.4e}, CosSim: {val_cossim:.4f}")

    if lp_args.save_dir:
        os.makedirs(lp_args.save_dir, exist_ok=True)
        util.save_pkl(lp_args.save_dir, "reg", reg)
        util.save_pkl(lp_args.save_dir, "metrics", dict(score=val_score, mse=val_mse, cossim=val_cossim))


if __name__ == "__main__":
    main(tyro.cli(LinearProbeArgs))
