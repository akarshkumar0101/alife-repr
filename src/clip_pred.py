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

from substrates.gol import GameOfLife, int2binary
from rollout import rollout_simulation
from foundation_models import create_foundation_model

import util

@dataclass
class OptimizerArgs:
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.
    clip_grad_norm: float = 1.0

@dataclass
class DataArgs:
    grid_size: int = 64 # grid size of the worlds for CLIP
    t_start: int = 128
    t_end: int = 1024 # timesteps are sampled from t_start to t_end
    batch_size: int = 256 # number of samples
    n_worlds: int | None = None # number of worlds to use for training MLP

@dataclass
class ModelArgs:
    d_hidden: int = 1024
    n_layers: int = 16

@dataclass
class Args:
    seed: int = 0
    save_dir: str | None = None
    cache_dataset_dir: str | None = "./data/"
    n_iters: int = 100000
    log_every: int = 1000

    model: ModelArgs = ModelArgs()
    opt: OptimizerArgs = OptimizerArgs()
    data: DataArgs = DataArgs()

class CLIPPredictor(nn.Module):
    cfg: ModelArgs

    @nn.compact
    def __call__(self, x):
        print('identity')
        x = int2binary(x).astype(float)
        x = nn.Dense(self.cfg.d_hidden, bias_init=nn.initializers.normal(0.01))(x)
        for _ in range(self.cfg.n_layers):
            identity = x
            x = nn.LayerNorm()(x)
            x = nn.Dense(self.cfg.d_hidden)(x)
            x = nn.gelu(x)
            x = nn.Dense(self.cfg.d_hidden)(x)
            x = x + identity
        x = nn.Dense(512, kernel_init=nn.initializers.normal(0.01))(x)
        return x

class Main:
    def __init__(self, args: Args):
        print(args)
        self.net = CLIPPredictor(args.model)
        self.substrate = GameOfLife(grid_size=args.data.grid_size)
        self.rollout_fn = partial(rollout_simulation, s0=None, substrate=self.substrate, fm=None, rollout_steps=args.data.t_end,
                                  time_sampling='video', img_size=None, return_state=True)
        self.args = args

        # self.generate_batch = jax.jit(self.generate_batch)
        self.do_iter_train = jax.jit(partial(self.do_iter, train=True))
        self.do_iter_eval = jax.jit(partial(self.do_iter, train=False))

        self.fm = create_foundation_model('clip')

        gol_data = np.load("./data/sweep_gol.npz", allow_pickle=True)
        self.params_all = jnp.array(gol_data['params']) # shape: (number of simulations, )
        self.oe_score_all = jnp.array(gol_data['oe_score']) # shape: (number of simulations, )

        rng = jax.random.PRNGKey(args.seed)
        self.n_worlds = self.args.data.n_worlds if self.args.data.n_worlds else len(self.params_all)
        idx = jax.random.permutation(rng, jnp.arange(self.n_worlds))
        n_train = int(self.n_worlds * 0.8)
        self.idx_train, self.idx_test = idx[:n_train], idx[n_train:]
    
    
    def create_clip_dataset(self):
        def get_img(rng, gol_params):
            rgb = self.rollout_fn(rng, gol_params)['rgb']
            t0 = jax.random.randint(rng, shape=(), minval=self.args.data.t_start, maxval=self.args.data.t_end)
            t0 = t0 - t0%2 # make sure t0 is even
            return rgb[t0]

        def get_avg_clip_vec(rng, gol_params):
            imgs = jax.vmap(get_img, in_axes=(0, None))(split(rng, self.args.data.batch_size), gol_params)
            z = jax.vmap(self.fm.embed_img)(imgs)
            z_avg = z.mean(axis=0)
            return dict(imgs=imgs, z=z, z_avg=z_avg)
        
        try:
            self.clip_dataset = util.load_pkl(self.args.cache_dataset_dir, "clip_dataset")
            print(f"Loaded cached dataset from {self.args.cache_dataset_dir}")
        except:
            print(f"No cached dataset found at {self.args.cache_dataset_dir}, creating new one")
            get_avg_clip_vec = jax.jit(get_avg_clip_vec)
            rng = jax.random.PRNGKey(0)
            self.clip_dataset = []
            for params in tqdm(self.params_all):
                self.clip_dataset.append(get_avg_clip_vec(rng, params)['z_avg'])
            self.clip_dataset = jnp.stack(self.clip_dataset)
            if self.args.cache_dataset_dir is not None:
                util.save_pkl(self.args.cache_dataset_dir, "clip_dataset", self.clip_dataset)

        self.clip_dataset = (self.clip_dataset - self.clip_dataset.mean(axis=0)) / (self.clip_dataset.std(axis=0) + 1e-8)

    def generate_instance(self, rng, train=True):
        idx = self.idx_train if train else self.idx_test
        ii = jax.random.randint(rng, shape=(), minval=0, maxval=len(idx))
        idx = idx[ii]
        gol_params = self.params_all[idx]
        clip_vec = self.clip_dataset[idx]
        return dict(gol_params=gol_params, clip_vec=clip_vec)

    def generate_batch(self, rng, train=True):
        return jax.vmap(partial(self.generate_instance, train=train))(split(rng, self.args.opt.batch_size))

    def loss_fn(self, params, batch):
        gol_params, clip_vec = batch['gol_params'], batch['clip_vec']
        pred = jax.vmap(self.net.apply, in_axes=(None, 0))(params, gol_params)
        loss = ((pred-clip_vec)**2).mean()
        metrics = dict(loss=loss, gol_params=gol_params, pred=pred, clip_vec=clip_vec)
        return loss, metrics

    def do_iter(self, train_state, rng, train=True):
        batch = self.generate_batch(rng, train=train)
        (_, metrics), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(train_state.params, batch)
        if train:
            train_state = train_state.apply_gradients(grads=grads)
        return train_state, {'grads': grads, **metrics}
    
    def init(self):
        self.create_clip_dataset()

        rng = jax.random.PRNGKey(self.args.seed)
        instance = self.generate_instance(rng)
        self.init_params = self.net.init(rng, instance['gol_params'])
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
        
        self.metrics = metrics
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
