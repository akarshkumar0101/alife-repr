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
class FeatureVizArgs:
    seed: int = 0
    load_dir: str | None = None
    save_dir: str | None = None

    n_iters: int = 2000

def main(fv_args: FeatureVizArgs):
    print(fv_args)
    args = util.load_pkl(fv_args.load_dir, "args")
    main = Main(args)
    net_params = util.load_pkl(args.save_dir, "params")

    def init_params(rng, grid_size=64):
        params = {}
        gs = 1
        while gs < grid_size:
            rng, _rng = split(rng)
            params[f'res_{gs:03d}'] = jax.random.normal(_rng, (gs, gs, 2))
            gs = gs * 2
        gs = grid_size
        rng, _rng = split(rng)
        params[f'res_{gs:03d}'] = jax.random.normal(_rng, (gs, gs, 2))
        return params

    def get_x0(rng, params, do_augs=True):
        gs = max([v.shape[0] for v in params.values()])
        x0 = jnp.zeros((gs, gs, 2))
        for k, v in params.items():
            x0 = x0 + jax.image.resize(v, (gs, gs, 2), method='bilinear')
        x0 = x0 + jax.random.normal(rng, x0.shape)*.1

        if do_augs:
            rng, _rng = split(rng)
            scale = jax.random.uniform(rng, (2,), minval=0.8, maxval=1.2)
            rng, _rng = split(rng)
            translate = jax.random.uniform(rng, (2,), minval=-gs//5, maxval=gs//5)
            x0 = jax.image.scale_and_translate(x0, (gs, gs, 2), (0, 1), scale, translate, method='bilinear')
        x0 = jax.nn.softmax(x0, axis=-1)
        return x0

    def loss_fn(params, rng, i_neuron):
        x0 = jax.vmap(get_x0, in_axes=(0, None))(split(rng, 4), params)
        dt_id = jnp.array(0, dtype=int)
        apply_fn = jax.vmap(partial(main.net.apply, dt_id=dt_id, return_hidden_reprs=True, method=main.net.forward_soft), in_axes=(None, 0))
        logits, hidden_reprs = apply_fn(net_params, x0)
        features = jax.tree.map(lambda x: x.mean(axis=(-3, -2)), hidden_reprs)
        features = jnp.concatenate(features, axis=-1)
        return features[..., i_neuron].mean()

    def do_iter(train_state, rng, i_neuron):
        loss, grads = jax.value_and_grad(loss_fn)(train_state.params, rng, i_neuron)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, loss

    @jax.jit
    def get_feature_viz(rng, i_neuron):
        params = init_params(rng)

        tx = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(1e-2, weight_decay=0., eps=1e-8))
        train_state = TrainState.create(apply_fn=main.net.apply, params=params, tx=tx)

        do_iter_fn = jax.jit(partial(do_iter, i_neuron=i_neuron))
        train_state, loss_history = jax.lax.scan(do_iter_fn, train_state, split(rng, fv_args.n_iters))
        return train_state.params, loss_history
    
    rng = jax.random.PRNGKey(fv_args.seed)
    params, loss_history = jax.lax.map(partial(get_feature_viz, rng), jnp.arange(0, 512, 4), batch_size=8)

    x0 = jax.vmap(partial(get_x0, do_augs=False), in_axes=(None, 0))(rng, params)
    print(x0.shape)
    if fv_args.save_dir:
        os.makedirs(fv_args.save_dir, exist_ok=True)
        util.save_pkl(fv_args.save_dir, "params", params)
        util.save_pkl(fv_args.save_dir, "loss_history", loss_history)
        util.save_pkl(fv_args.save_dir, "x0", x0)

if __name__ == "__main__":
    main(tyro.cli(FeatureVizArgs))