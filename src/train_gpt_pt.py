import time
import math
import os
import copy
from functools import partial
from dataclasses import dataclass
from tqdm import tqdm
import tyro
import random

import numpy as np
# import jax
# import jax.numpy as jnp
# from jax.random import split

import torch
from x_transformers import AutoregressiveWrapper, TransformerWrapper, Decoder
from einops import rearrange

# from substrates.gol import GameOfLife
# from rollout import rollout_simulation
from substrates.gol_pt import run_game_of_life
import util

@dataclass
class DataArgs:
    gol_params: int = 6152
    t_start: int = 32
    t_end: int = 64
    grid_size: int = 30
    timesteps: int = 10
    dt: int = 1
    patch_size: int = 3

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

    decay_lr: bool | None = False

@dataclass
class ModelArgs:
    # n_layer: int = 12
    n_layer: int = 6
    d_model: int = 512
    n_heads: int = 8
    vocab_size: int = 512
    ctx_len: int = 1024
    rotary_pos_emb: bool | None = False

@dataclass
class Args:
    seed: int = 0
    save_dir: str | None = None
    n_iters: int = 10000
    log_every: int = 1000

    model: ModelArgs = ModelArgs()
    data: DataArgs = DataArgs()
    opt: OptimizerArgs = OptimizerArgs()

class DataGenerator:
    def __init__(self, args: DataArgs):
        self.args = copy.deepcopy(args)
        # self.substrate = GameOfLife(grid_size=self.args.grid_size)
        # self.rollout_fn = partial(rollout_simulation, s0=None, substrate=self.substrate, fm=None, rollout_steps=self.args.t_end+self.args.timesteps,
                                #   time_sampling='video', img_size=None, return_state=True)
        self.params = torch.tensor(self.args.gol_params).cuda()
    
    # def generate_instance(self, rng):
    #     rng, _rng = split(rng)
    #     state = self.rollout_fn(_rng, self.args.gol_params)['state']
    #     rng, _rng = split(rng)
    #     t0 = jax.random.randint(_rng, shape=(), minval=self.args.t_start, maxval=self.args.t_end)
    #     data = jax.lax.dynamic_slice(state, (t0, 0, 0), (self.args.timesteps, self.args.grid_size, self.args.grid_size))
    #     data = data[::self.args.dt]
    #     data = rearrange(data, "T (H h) (W w) -> (T H W) (h w)", h=self.args.patch_size, w=self.args.patch_size)
    #     data = (data * (2**jnp.arange(self.args.patch_size**2))).sum(axis=-1)
    #     return dict(rng=rng, state=state, t0=t0, data=data)

    def generate_batch(self, rng, batch_size):
        # return jax.vmap(self.generate_instance)(split(rng, batch_size))
        state = run_game_of_life(self.params, batch_size=batch_size, grid_size=self.args.grid_size, t_steps=self.args.t_end+self.args.timesteps)
        t0 = torch.randint(self.args.t_start, self.args.t_end, (batch_size,))
        data = torch.stack([s[t0_: t0_ + self.args.timesteps] for s, t0_ in zip(state, t0)])
        data = data[:, ::self.args.dt]
        data = rearrange(data, "B T (H h) (W w) -> B (T H W) (h w)", h=self.args.patch_size, w=self.args.patch_size)
        data = (data * (2**torch.arange(self.args.patch_size**2, device=data.device))).sum(axis=-1).long()
        return dict(state=state, data=data)


class Main:
    def __init__(self, args: Args):
        self.args = copy.deepcopy(args)

        self.data_gen = DataGenerator(self.args.data)
        self.data_gen.generate_batch = partial(self.data_gen.generate_batch, batch_size=self.args.opt.batch_size)
        # self.data_time, self.model_time = 0., 0.

    def do_iter_train(self, rng):
        # data_time_start = time.time()
        batch = self.data_gen.generate_batch(rng)['data']
        # batch = torch.from_numpy(np.array(batch)).cuda().long()
        batch = rearrange(batch, "(N B) ... -> N B ...", N=self.args.opt.grad_accum_steps)
        # data_time = time.time() - data_time_start
        # model_time_start = time.time()
        loss = 0.
        for mbatch in batch:
            mloss = self.model(mbatch) / self.args.opt.grad_accum_steps
            mloss.backward()
            loss += mloss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.opt.clip_grad_norm)
        self.opt.step()
        self.opt.zero_grad()
        # model_time = time.time() - model_time_start
        # self.data_time += data_time
        # self.model_time += model_time
        # print(f"Data time: {self.data_time:.2f}s, Model time: {self.model_time:.2f}s")
        return dict(loss=loss.item(), grad_norm=grad_norm.item())
    
    @torch.no_grad()
    def do_iter_eval(self, rng):
        batch = self.data_gen.generate_batch(rng)['data']
        # batch = torch.from_numpy(np.array(batch)).cuda().long()
        batch = rearrange(batch, "(N B) ... -> N B ...", N=self.args.opt.grad_accum_steps)
        loss = 0.
        for mbatch in batch:
            mloss = self.model(mbatch) / self.args.opt.grad_accum_steps
            loss += mloss.item()
        return dict(loss=loss, grad_norm=None)

    def get_lr(self, it):
        warmup_iters = self.args.n_iters//100
        if it < warmup_iters:
            return self.args.opt.learning_rate * it / warmup_iters
        else:
            if not self.args.opt.decay_lr:
                return self.args.opt.learning_rate
            else:
                decay_ratio = (it - warmup_iters) / (self.args.n_iters - warmup_iters)
                assert 0 <= decay_ratio <= 1
                coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
                learning_rate, min_lr = self.args.opt.learning_rate, self.args.opt.learning_rate/10.
                return min_lr + coeff * (learning_rate - min_lr)

    def init(self):
        self.model = Decoder(dim=self.args.model.d_model, depth=self.args.model.n_layer, heads=self.args.model.n_heads, rotary_pos_emb=self.args.model.rotary_pos_emb)
        self.model = TransformerWrapper(num_tokens=self.args.model.vocab_size, max_seq_len=self.args.model.ctx_len, attn_layers=self.model)
        self.model = AutoregressiveWrapper(self.model).cuda()
        # self.model = torch.compile(self.model)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters: {n_params:,}")

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.args.opt.learning_rate,
                                     betas=(self.args.opt.beta1, self.args.opt.beta2), eps=self.args.opt.eps, weight_decay=self.args.opt.weight_decay)

        self.final_loss = np.inf
        self.loss_history, self.grad_norm_history = [], []

        # self.rng = jax.random.PRNGKey(self.args.seed)
        self.i_iter = 0
        self.start_time = time.time()


        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
    
    def step(self):
        lr = self.get_lr(self.i_iter)
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

        # self.rng, _rng = split(self.rng)
        metrics = self.do_iter_train(None)

        self.loss_history.append(metrics['loss'])
        self.grad_norm_history.append(metrics['grad_norm'])
        if self.args.save_dir is not None and (self.i_iter % self.args.log_every == 0 or self.i_iter == self.args.n_iters - 1):
            os.makedirs(self.args.save_dir, exist_ok=True)
            util.save_pkl(self.args.save_dir, "args", self.args)
            util.save_pkl(self.args.save_dir, "loss_history", self.loss_history)
            util.save_pkl(self.args.save_dir, "grad_norm_history", self.grad_norm_history)
            iter_per_sec = self.i_iter / (time.time() - self.start_time)
            util.save_pkl(self.args.save_dir, "iter_per_sec", iter_per_sec)

            final_loss_now = np.mean([self.do_iter_eval(None)['loss'] for _ in range(100)])
            if final_loss_now < self.final_loss:
                self.final_loss = final_loss_now
                util.save_pkl(self.args.save_dir, "final_loss", self.final_loss)
                # util.save_pkl(self.args.save_dir, "params", jax.tree.map(lambda x: np.array(x), self.train_state.params))
                torch.save(self.model, f"{self.args.save_dir}/model.pt")
        
        self.i_iter += 1
    
    def run(self):
        self.init()
        pbar = tqdm(range(self.args.n_iters))
        for _ in pbar:
            self.step()
            pbar.set_postfix(loss=self.loss_history[-1], ppl=np.exp(self.loss_history[-1]))

def main():
    args = tyro.cli(Args)
    print(args)
    main = Main(args)
    main.run()

if __name__ == "__main__":
    main()
