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

from substrates.gol import GameOfLife, int2binary
from rollout import rollout_simulation
import util


@dataclass
class OptimizerArgs:
    batch_size:    int = 256
    learning_rate: float = 3e-4
    weight_decay:  float = 0.
    clip_grad_norm: float = 1.0

@dataclass
class DataArgs:
    grid_size:        int = 64
    t_start:          int = 128
    t_end:            int = 1024
    batch_size:       int = 256
    n_worlds:        int | None = None
    frames_per_param: int = 4

@dataclass
class ModelArgs:
    d_hidden: int = 128

@dataclass
class Args:
    seed:               int = 0
    save_dir:          str | None = None
    cache_dataset_dir: str | None = "./data/"
    n_iters:            int = 100000
    log_every:          int = 1000

    model: ModelArgs     = field(default_factory=ModelArgs)
    opt:   OptimizerArgs = field(default_factory=OptimizerArgs)
    data:  DataArgs      = field(default_factory=DataArgs)


class ParamEncoder(nn.Module):
    d_hidden: int
    @nn.compact
    def __call__(self, params):
        x = int2binary(params).astype(float)
        x = nn.Dense(self.d_hidden)(x); x = nn.gelu(x)
        return nn.Dense(self.d_hidden)(x)

class TimeEncoder(nn.Module):
    d_hidden: int
    @nn.compact
    def __call__(self, tau):
        t = tau[..., None]
        x = nn.Dense(self.d_hidden)(t); return nn.gelu(x)

class ConvBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.out_channels, (3,3), padding="SAME")(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.gelu(x)
        x = nn.Conv(self.out_channels, (3,3), padding="SAME")(x)
        x = nn.GroupNorm(num_groups=8)(x)
        return nn.gelu(x)

class Down(nn.Module):
    out_ch: int

    @nn.compact
    def __call__(self, x):
        # convs at full resolution
        x = ConvBlock(self.out_ch)(x)
        skip = x                            # save before pooling
        # spatial downsample by 2
        x = nn.max_pool(x, (2,2), (2,2))
        return x, skip

class Up(nn.Module):
    out_ch: int

    @nn.compact
    def __call__(self, x, skip):
        # x: [B,H',W',C], skip: [B,2H',2W',C_skip]
        b, h, w, c = x.shape
        # upsample by exactly factor 2
        x = jax.image.resize(x, (b, h*2, w*2, c), method="nearest")
        # now x.shape == skip.shape[:-1] + (c,)
        x = jnp.concatenate([x, skip], axis=-1)
        return ConvBlock(self.out_ch)(x)

class FlowUNet(nn.Module):
    base_ch: int = 64
    depth:   int = 4  # number of down/up steps

    @nn.compact
    def __call__(self, x):
        skips = []
        # encoder
        for i in range(self.depth):
            ch = self.base_ch * (2**i)
            x, skip = Down(ch)(x)
            skips.append(skip)
        # bottleneck
        x = ConvBlock(self.base_ch * (2**self.depth))(x)
        # decoder
        for i in reversed(range(self.depth)):
            ch   = self.base_ch * (2**i)
            skip = skips.pop()
            x    = Up(ch)(x, skip)
        # project to 3‑channel velocity
        return nn.Conv(3, (1,1), padding="SAME")(x)


def noise_and_velocity_fn(x0, tau, key):
    eps = jax.random.normal(key, x0.shape)
    x_tau = x0 + tau[..., None, None, None] * eps
    return x_tau, eps


class Main:
    def __init__(self, args: Args):
        print(args)
        self.args = args

        self.param_enc = ParamEncoder(d_hidden=args.model.d_hidden)
        self.time_enc  = TimeEncoder(d_hidden=args.model.d_hidden)
        self.net       = FlowUNet()

        self.substrate = GameOfLife(grid_size=args.data.grid_size)
        self.rollout_fn = partial(
            rollout_simulation,
            s0=None, substrate=self.substrate,
            fm=None, rollout_steps=args.data.t_end,
            time_sampling='video', img_size=None,
            return_state=True
        )

        self.do_iter_train = jax.jit(partial(self.do_iter, train=True))
        self.do_iter_eval  = jax.jit(partial(self.do_iter, train=False))

        # Load all parameters from data file
        gol_data = np.load("./data/sweep_gol.npz", allow_pickle=True)
        self.params_all = jnp.array(gol_data['params'])  # ALL parameters from file
        
        # Determine subset for training
        rng = jax.random.PRNGKey(args.seed)
        self.n_worlds = self.args.data.n_worlds if self.args.data.n_worlds else len(self.params_all)
        idx = jax.random.permutation(rng, jnp.arange(self.n_worlds))
        n_train = int(self.n_worlds * 0.8)
        self.idx_train, self.idx_test = idx[:n_train], idx[n_train:]


    def create_frame_dataset(self):
        def get_img(rng, gol_params):
            rgb = self.rollout_fn(rng, gol_params)['rgb']
            t0 = jax.random.randint(rng, shape=(), minval=self.args.data.t_start, maxval=self.args.data.t_end)
            t0 = t0 - t0 % 2  # make sure t0 is even
            return rgb[t0]

        def get_frames(rng, gol_params):
            imgs = jax.vmap(get_img, in_axes=(0, None))(split(rng, self.args.data.frames_per_param), gol_params)
            return imgs

        try:
            raw_dataset = util.load_pkl(self.args.cache_dataset_dir, "frame_dataset")
            print(f"Loaded cached dataset from {self.args.cache_dataset_dir}")
        except:
            print(f"No cached dataset found at {self.args.cache_dataset_dir}, creating new one for ALL parameters")
            get_frames = jax.jit(get_frames)
            rng = jax.random.PRNGKey(0)
            raw_dataset = []
            for params in tqdm(self.params_all):  # Process ALL parameters
                frames = get_frames(rng, params)
                raw_dataset.append(frames)
                rng, _ = split(rng)  # advance rng for next parameter
            raw_dataset = jnp.stack(raw_dataset)  # Shape: [n_all_params, frames_per_param, H, W, C]
            if self.args.cache_dataset_dir is not None:
                os.makedirs(self.args.cache_dataset_dir, exist_ok=True)
                util.save_pkl(self.args.cache_dataset_dir, "frame_dataset", raw_dataset)

        # Normalize using all frames
        self.norm_mean = raw_dataset.mean()
        self.norm_std = raw_dataset.std() + 1e-8
        print(f"Computed normalization mean={self.norm_mean}, std={self.norm_std}")
        
        # Extract subset for current training (first n_worlds parameters)
        self.frame_dataset = raw_dataset[:self.n_worlds]  # Shape: [n_worlds, frames_per_param, H, W, C]
        self.frame_dataset = (self.frame_dataset - self.norm_mean) / self.norm_std


    def generate_instance(self, rng, train=True):
        idx = self.idx_train if train else self.idx_test
        ii = jax.random.randint(rng, shape=(), minval=0, maxval=len(idx))
        world_idx = idx[ii]
        
        # Sample a random frame from this world
        rng, sub = split(rng)
        frame_idx = jax.random.randint(sub, shape=(), minval=0, maxval=self.args.data.frames_per_param)
        
        gol_params = self.params_all[world_idx]
        target_img = self.frame_dataset[world_idx, frame_idx]
        
        return dict(gol_params=gol_params, target_img=target_img)


    def generate_batch(self, rng, train=True):
        rngs = split(rng, self.args.opt.batch_size)
        return jax.vmap(partial(self.generate_instance, train=train))(rngs)


    def loss_fn(self, params, batch, taus, rng):
        gol_params, target_img = batch['gol_params'], batch['target_img']
        # split rng to generate per-example keys
        rng, k1 = split(rng)
        keys = jax.random.split(k1, target_img.shape[0])
        # compute noised frames & true velocities
        x_tau, v_true = jax.vmap(noise_and_velocity_fn)(target_img, taus, keys)
        e     = jax.vmap(self.param_enc.apply, in_axes=(None,0))(params['param_enc'], gol_params)
        t_emb = jax.vmap(self.time_enc.apply,  in_axes=(None,0))(params['time_enc'],  taus)
        H, W, _ = x_tau.shape[1:]
        cond = jnp.concatenate([e, t_emb], axis=-1)
        cond = cond[:,None,None,:].repeat(H,1).repeat(W,2)
        inp  = jnp.concatenate([x_tau, cond], axis=-1)
        v_hat = self.net.apply(params['unet'], inp)
        loss  = ((v_hat - v_true)**2).mean()
        metrics = dict(loss=loss, gol_params=gol_params, pred_v=v_hat, true_v=v_true)
        return loss, (metrics, rng)


    def do_iter(self, train_state, rng, train=True):
        rng, sub = split(rng)
        taus = jax.random.uniform(sub, (self.args.opt.batch_size,), minval=0., maxval=1.)
        rng, bsub = split(rng)
        batch = self.generate_batch(bsub, train=train)
        (loss, (metrics, new_rng)), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(train_state.params, batch, taus, rng)
        if train:
            new_state = train_state.apply_gradients(grads=grads)
        else:
            new_state = train_state
        return new_state, new_rng, {'grads': grads, **metrics}


    def init(self):
        self.create_frame_dataset()
        
        rng = jax.random.PRNGKey(self.args.seed)
        self.rng = rng
        rng, sub = split(self.rng)
        batch = self.generate_batch(sub, train=True)
        x0 = batch['target_img']; p0 = batch['gol_params'][0]
        rng, sub = split(rng)
        tau0 = jax.random.uniform(sub, (), minval=0., maxval=1.)
        x_tau0, _ = noise_and_velocity_fn(x0[0], tau0, split(rng)[0])
        pe = self.param_enc.init(rng, p0)
        te = self.time_enc.init(rng, tau0)
        e = self.param_enc.apply(pe, p0)
        t_emb = self.time_enc.apply(te, tau0)
        H, W, _ = x_tau0.shape
        cond = jnp.concatenate([e, t_emb], axis=-1)
        cond_map = jnp.tile(cond[None, None, :], (H, W, 1))
        inp = jnp.concatenate([x_tau0, cond_map], axis=-1)
        inp = inp[None, ...]
        un = self.net.init(rng, inp)
        self.init_params = {'param_enc': pe, 'time_enc': te, 'unet': un}
        tx = optax.chain(
            optax.clip_by_global_norm(self.args.opt.clip_grad_norm),
            optax.adamw(self.args.opt.learning_rate, weight_decay=self.args.opt.weight_decay, eps=1e-8)
        )
        self.train_state = TrainState.create(
            apply_fn=None,
            params=self.init_params,
            tx=tx
        )
        self.loss_history = []
        self.final_loss = jnp.inf
        self.i_iter = 0


    def step(self):
        self.rng, _rng = split(self.rng)
        self.train_state, self.rng, metrics = self.do_iter_train(self.train_state, _rng)

        self.loss_history.append(metrics['loss'].item())
        if self.args.save_dir is not None and (self.i_iter % self.args.log_every == 0 or self.i_iter == self.args.n_iters - 1):
            os.makedirs(self.args.save_dir, exist_ok=True)
            util.save_pkl(self.args.save_dir, "args", self.args)
            util.save_pkl(self.args.save_dir, "loss_history", self.loss_history)
            eval_losses = []
            for _ in range(10):  # Evaluate on 10 batches
                eval_key, self.rng = split(self.rng)
                _, eval_key, eval_metrics = self.do_iter_eval(self.train_state, eval_key)
                eval_losses.append(eval_metrics['loss'].item())
            final_loss_now = np.mean(eval_losses)
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
            pbar.set_postfix(loss=self.loss_history[-1])
    

    def sample(self, params, gol_param_int, rng, n_steps=100):
        taus = list(np.linspace(1.0, 0.0, n_steps, dtype=np.float32))

        # initialize x
        x = jax.random.normal(rng, (self.args.data.grid_size, self.args.data.grid_size, 3))
        H, W, _ = x.shape

        # encode params
        e = self.param_enc.apply(params['param_enc'], gol_param_int)
        for t0, t1 in zip(taus[:-1], taus[1:]):
            tau_mid = 0.5 * (t0 + t1)
            t_emb   = self.time_enc.apply(params['time_enc'], tau_mid)
            cond    = jnp.concatenate([e, t_emb], axis=-1)
            cond_map= cond[None, None, None, :].repeat(H, 1).repeat(W, 2)
            inp     = jnp.concatenate([x[None], cond_map], axis=-1)
            v       = self.net.apply(params['unet'], inp)[0]
            # Euler update
            x       = x - (t0 - t1) * v
        return x * self.norm_std + self.norm_mean



if __name__ == "__main__":
    main = Main(tyro.cli(Args))
    main.init()
    main.run()
