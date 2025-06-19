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

    @jax.jit
    def get_features(params, batch):
        x0, x1, dt_id = batch['x0'], batch['x1'], batch['dt_id']
        forward_fn = jax.vmap(partial(main.net.apply, return_hidden_reprs=True), in_axes=(None, 0, 0))
        y, hidden_reprs = forward_fn(params, x0, dt_id)
        features = jax.tree.map(lambda x: x.mean(axis=(-3, -2)), hidden_reprs)
        features = jnp.concatenate(features, axis=-1)
        return features

    @jax.jit
    def get_target(batch):
        x = batch[lp_args.target.lower()] # x0 or x1
        render_fn = jax.vmap(partial(main.substrate.render_state, params=args.data.gol_params, img_size=224))
        img = render_fn(x)
        z = jax.vmap(fm.embed_img)(img)
        return z
    
    rng = jax.random.PRNGKey(lp_args.seed)
    if lp_args.ablate_features.lower() == "random":
        instance = main.generate_instance(rng)
        params = main.net.init(rng, instance['x0'], instance['dt_id'])
    else:
        params = util.load_pkl(lp_args.load_dir, "params")

    X, Y = [], []
    pbar = tqdm(range(lp_args.n_iters))
    for _ in pbar:
        rng, _rng = split(rng)
        batch = main.generate_batch(_rng)
        X.append(get_features(params, batch))
        Y.append(get_target(batch))
    X = jnp.concatenate(X, axis=0)
    Y = jnp.concatenate(Y, axis=0)

    if lp_args.ablate_features.lower() == "zeros":
        X = jnp.zeros_like(X)

    X, Y = np.array(X), np.array(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=lp_args.seed)
    print(f"Training: {X_train.shape} -> {Y_train.shape}")
    print(f"Testing: {X_test.shape} -> {Y_test.shape}")
    reg = LinearRegression(fit_intercept=True).fit(X_train, Y_train)
    score_train = reg.score(X_train, Y_train)
    score_test = reg.score(X_test, Y_test)
    Y_train_pred = reg.predict(X_train)
    Y_test_pred = reg.predict(X_test)
    mse_train = ((Y_train-Y_train_pred)**2).mean()
    mse_test = ((Y_test-Y_test_pred)**2).mean()
    Y_train_pred_norm = Y_train_pred / (jnp.linalg.norm(Y_train_pred, axis=-1, keepdims=True) + 1e-8)
    Y_test_pred_norm = Y_test_pred / (jnp.linalg.norm(Y_test_pred, axis=-1, keepdims=True) + 1e-8)
    cossim_train = (Y_train*Y_train_pred_norm).sum(axis=-1).mean()
    cossim_test = (Y_test*Y_test_pred_norm).sum(axis=-1).mean()
    metrics = dict(score_train=score_train, score_test=score_test, mse_train=mse_train, mse_test=mse_test,
                   cossim_train=cossim_train, cossim_test=cossim_test)
    print(metrics)

    if lp_args.save_dir:
        os.makedirs(lp_args.save_dir, exist_ok=True)
        util.save_pkl(lp_args.save_dir, "reg", reg)
        util.save_pkl(lp_args.save_dir, "metrics", metrics)

if __name__ == "__main__":
    main(tyro.cli(LinearProbeArgs))