import time
import os
import copy
from functools import partial
from dataclasses import dataclass
from tqdm import tqdm
import tyro

import numpy as np
import jax
import jax.numpy as jnp
from jax.random import split
import optax
from flax.training.train_state import TrainState
from flax import linen as nn
from einops import rearrange

from substrates.gol import GameOfLife
from rollout import rollout_simulation

import util

@dataclass
class ModelArgs:
    d_hidden: int = 1024
    layers: int = 8

@dataclass
class DataArgs:
    gol_params: int = 6152
    grid_size: int = 64
    dt: int = 32
    t_start: int = 32
    t_end: int = 64

@dataclass
class OptimizerArgs:
    batch_size: int = 32
    grad_accum_steps: int = 1
    clip_grad_norm: float = 1.0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    decay_lr: bool | None = True

@dataclass
class Args:
    seed: int = 0
    save_dir: str | None = None
    n_iters: int = 10000
    log_every: int = 1000

    model: ModelArgs = ModelArgs()
    data: DataArgs = DataArgs()
    opt: OptimizerArgs = OptimizerArgs()

class Model(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        x, y = inputs['x'], inputs['y']
        y_pred = nn.Dense(features=len(y))(x)
        loss = ((y_pred - y)**2).mean()
        outputs = dict(loss=loss, y_pred=y_pred, x=x, y=y)
        return outputs

class DataGenerator:
    def __init__(self, args: DataArgs):
        self.args = copy.deepcopy(args)
        self.substrate = GameOfLife(grid_size=self.args.grid_size)
        self.rollout_fn = partial(rollout_simulation, s0=None, substrate=self.substrate, fm=None, rollout_steps=self.args.t_end+self.args.dt,
                                  time_sampling='video', img_size=None, return_state=True)
    
    def generate_instance(self, rng):
        rng, _rng = split(rng)
        state = self.rollout_fn(_rng, self.args.data.gol_params)['state']
        rng, _rng = split(rng)
        t0 = jax.random.randint(_rng, shape=(), minval=self.args.data.t_start, maxval=self.args.data.t_end)
        t1 = t0 + self.args.data.dt
        state = jax.lax.dynamic_slice(state, (t0, 0, 0), (self.args.data.dt, self.args.data.grid_size, self.args.data.grid_size))
        return dict(rng=rng, state=state, t0=t0, t1=t1)

    def generate_batch(self, rng):
        return jax.vmap(self.generate_instance)(split(rng, self.args.opt.batch_size))

class Main:
    def __init__(self, args: Args):
        self.args = copy.deepcopy(args)

        self.model = Model(self.args.model)
        self.data_gen = DataGenerator(self.args.data)
        self.do_iter_train = jax.jit(self.do_iter_train)
        self.do_iter_eval = jax.jit(self.do_iter_eval)

    def loss_fn(self, params, batch):
        inputs = dict(x=batch['state'], y=batch['state'])
        outputs = jax.vmap(self.model.apply, in_axes=(None, 0, 0))(params, inputs)
        return outputs['loss'].mean(), outputs
    
    def do_iter(self, train_state, rng, train=True):
        batch = self.data_gen.generate_batch(rng)
        batch = jax.tree.map(lambda x: rearrange(x, "(N B) ... -> N B ...", N=self.args.opt.grad_accum_steps), batch)
        def micro_step(grads, batch):
            (_, outputs), grads_new = jax.value_and_grad(self.loss_fn, has_aux=True)(train_state.params, batch)
            grads = jax.tree.map(jnp.add, grads, grads_new)
            return grads, outputs
        grads = jax.tree.map(lambda x: jnp.zeros_like(x), train_state.params)
        grads, outputs = jax.lax.scan(micro_step, grads, batch)
        grads = jax.tree.map(lambda x: x / self.args.opt.grad_accum_steps, grads)
        outputs = jax.tree.map(lambda x: rearrange(x, "N B ... -> (N B) ..."), outputs)
        grad_norm = jnp.linalg.norm(jnp.concatenate([g.flatten() for g in jax.tree.leaves(grads)]))
        if train:
            train_state = train_state.apply_gradients(grads=grads)
        return train_state, dict(loss=outputs['loss'].mean(), grad_norm=grad_norm if train else None)
    
    def do_iter_eval(self, train_state, rng):
        batch = self.data_gen.generate_batch(rng)
        batch = jax.tree.map(lambda x: rearrange(x, "(N B) ... -> N B ...", N=self.args.opt.grad_accum_steps), batch)
        def micro_step(_, batch):
            loss, outputs = self.loss_fn(train_state.params, batch)
            return None, outputs
        _, outputs = jax.lax.scan(micro_step, None, batch)
        outputs = jax.tree.map(lambda x: rearrange(x, "N B ... -> (N B) ..."), outputs)
        return train_state, dict(loss=outputs['loss'].mean(), grad_norm=None)

    def init(self):
        # self.clip = CLIP()
        rng = jax.random.PRNGKey(self.args.seed)
        instance = self.data_gen.generate_instance(rng)
        print(self.model.tabulate(rng, rng, instance['state']))
        self.init_params = self.model.init(rng, rng, instance['state'])
        n_params = sum(x.size for x in jax.tree.leaves(self.init_params))
        print(f"Number of parameters: {n_params:,}")

        lr_final = self.args.opt.learning_rate/10 if self.args.opt.decay_lr else self.args.opt.learning_rate
        lr_schedule = optax.warmup_cosine_decay_schedule(init_value=0., peak_value=self.args.opt.learning_rate,
                                                         warmup_steps=self.args.n_iters//100,
                                                         decay_steps=self.args.n_iters-self.args.n_iters//100,
                                                         end_value=lr_final, exponent=1.)
        tx = optax.chain(optax.clip_by_global_norm(self.args.opt.clip_grad_norm),
                         optax.adamw(learning_rate=lr_schedule, weight_decay=self.args.opt.weight_decay,
                                     b1=self.args.opt.beta1, b2=self.args.opt.beta2, eps=self.args.opt.eps))
        self.train_state = TrainState.create(apply_fn=self.model.apply, params=self.init_params, tx=tx)

        self.final_loss = np.inf
        self.loss_history, self.grad_norm_history = [], []

        self.rng = jax.random.PRNGKey(self.args.seed)
        self.i_iter = 0
        self.start_time = time.time()
    
    def step(self):
        self.rng, _rng = split(self.rng)
        self.train_state, metrics = self.do_iter_train(self.train_state, _rng)

        self.loss_history.append(metrics['loss'].mean().item())
        self.grad_norm_history.append(metrics['grad_norm'].item())
        if self.args.save_dir is not None and (self.i_iter % self.args.log_every == 0 or self.i_iter == self.args.n_iters - 1):
            os.makedirs(self.args.save_dir, exist_ok=True)
            util.save_pkl(self.args.save_dir, "args", self.args)
            util.save_pkl(self.args.save_dir, "loss_history", self.loss_history)
            util.save_pkl(self.args.save_dir, "grad_norm_history", self.grad_norm_history)
            iter_per_sec = self.i_iter / (time.time() - self.start_time)
            util.save_pkl(self.args.save_dir, "iter_per_sec", iter_per_sec)

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

def main():
    args = tyro.cli(Args)
    print(args)
    main = Main(args)
    main.init()
    main.run()

if __name__ == "__main__":
    main()
