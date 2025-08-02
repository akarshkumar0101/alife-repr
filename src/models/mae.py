import math
from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange, repeat

from .transformer import Block

from dataclasses import dataclass

import optax

@dataclass
class GPTConfig:
    n_layer: int = 4 # number of layers
    n_head: int = 8 # number of attention heads
    n_embd: int = 64 # embedding dimension
    dropout: float = 0. # dropout probability

@dataclass
class MAEConfig:
    tensor_shape: Tuple[int, int, int] = (4, 32, 32) # (T, H, W)
    patch_shape: Tuple[int, int, int] = (1, 4, 4) # (pt, ph, pw)
    mask_ratio: float = 0.1 # ratio of patches to mask
    enc: GPTConfig = GPTConfig(n_layer=12, n_head=8, n_embd=256) # encoder config
    dec: GPTConfig = GPTConfig(n_layer=4, n_head=4, n_embd=64) # decoder config


class Embed3D(nn.Module):
    nt: int
    nh: int
    nw: int
    features: int

    @nn.compact
    def __call__(self, x):
        xt, xh, xw = jnp.unravel_index(x, (self.nt, self.nh, self.nw))
        xt = nn.Embed(self.nt, self.features)(xt)
        xh = nn.Embed(self.nh, self.features)(xh)
        xw = nn.Embed(self.nw, self.features)(xw)
        x = xt + xh + xw
        return x

class MAE(nn.Module):
    cfg: MAEConfig

    def setup(self):
        cfg = self.cfg
        (T, H, W), (pt, ph, pw) = cfg.tensor_shape, cfg.patch_shape
        nt, nh, nw = T//pt, H//ph, W//pw
        self.tensor_dim, self.patch_dim = math.prod(cfg.tensor_shape), math.prod(cfg.patch_shape)
        self.n_patches = self.tensor_dim // self.patch_dim
        self.keep_len = int((1-cfg.mask_ratio) * self.n_patches)
        print(f"ctx_len: {self.n_patches}")
        print(f"patch_dim: {self.patch_dim}")

        # encoder
        self.patch_embed = nn.Dense(cfg.enc.n_embd, bias_init=nn.initializers.normal(0.01))
        # self.pos_embed_enc = nn.Embed(self.n_patches, cfg.enc.n_embd)
        self.pos_embed_enc = Embed3D(nt, nh, nw, cfg.enc.n_embd)
        self.blocks_enc = [Block(cfg.enc) for _ in range(cfg.enc.n_layer)]
        # decoder
        self.enc_to_dec = nn.Dense(cfg.dec.n_embd)
        self.dec_mask_token = nn.Embed(1, cfg.dec.n_embd)
        # self.pos_embed_dec = nn.Embed(self.n_patches, cfg.dec.n_embd)
        self.pos_embed_dec = Embed3D(nt, nh, nw, cfg.dec.n_embd)
        self.blocks_dec = [Block(cfg.dec) for _ in range(cfg.dec.n_layer)]
        self.head = nn.Dense(self.patch_dim)
    
    def __call__(self, rng: jax.random.PRNGKey, x: jax.Array, *, train: bool = True) -> jax.Array:
        """
        x.shape: (T, H, W)
        """
        cfg = self.cfg
        (T, H, W), (pt, ph, pw) = cfg.tensor_shape, cfg.patch_shape
        nt, nh, nw = T//pt, H//ph, W//pw
        x_in = x
        x = x.astype(float)
        # convert to flattened patches
        x = rearrange(x, "(nt pt) (nh ph) (nw pw) -> (nt nh nw) (pt ph pw)",
                      nt=nt, nh=nh, nw=nw, pt=pt, ph=ph, pw=pw)
        # mask patches
        idx = jax.random.permutation(rng, self.n_patches)
        idx_keep, idx_mask = idx[:self.keep_len], idx[self.keep_len:]
        x = x[idx_keep]

        input_mask = jnp.ones((self.n_patches,), dtype=int)
        input_mask = input_mask.at[idx_keep].set(0) # (nt*nh*nw,)
        input_mask = repeat(input_mask, "(nt nh nw) -> (nt pt) (nh ph) (nw pw)",
                            nt=nt, nh=nh, nw=nw, pt=pt, ph=ph, pw=pw)

        # embed patches
        x = self.patch_embed(x)
        # add positional embeddings
        x = x + self.pos_embed_enc(idx_keep)
        # forward encoder
        features = []
        for block in self.blocks_enc:
            x = rearrange(x, "... -> 1 ...")
            x = block(x, train=train)
            x = rearrange(x, "1 ... -> ...")
            features.append(x)
        features = jnp.stack(features)
        x_enc = x
        # convert to decoder dim
        x = self.enc_to_dec(x)
        # add mask tokens
        xp = self.dec_mask_token(jnp.zeros(self.n_patches, dtype=int))
        x = xp.at[idx_keep].set(x)
        # add positional embeddings
        x = x + self.pos_embed_dec(jnp.arange(self.n_patches))
        # forward decoder
        for block in self.blocks_dec:
            x = rearrange(x, "... -> 1 ...")
            x = block(x, train=train)
            x = rearrange(x, "1 ... -> ...")
        # project to patches
        x = self.head(x)
        # unflatten patches
        x = rearrange(x, "(nt nh nw) (pt ph pw) -> (nt pt) (nh ph) (nw pw)",
                      nt=nt, nh=nh, nw=nw, pt=pt, ph=ph, pw=pw)
        logits = x
        loss_ce = optax.losses.sigmoid_binary_cross_entropy(logits, x_in)
        loss = (loss_ce * input_mask).sum() / input_mask.sum()
        outputs = dict(x_in=x_in, idx_keep=idx_keep, idx_mask=idx_mask, input_mask=input_mask,
                       features=features, x_enc=x_enc, logits=logits, loss_ce=loss_ce, loss=loss)
        return outputs
    
def main():
    from tqdm.auto import tqdm
    from jax.random import split
    cfg = MAEConfig()
    net = MAE(cfg)
    rng = jax.random.PRNGKey(0)

    x = jnp.zeros((cfg.nt*cfg.pt, cfg.nh*cfg.ph, cfg.nw*cfg.pw), dtype=int)

    params = net.init(rng, rng, x)
    print(sum(x.size for x in jax.tree.leaves(params)))

    loss, metrics = net.apply(params, rng, x, train=True)
    print(jax.tree.map(lambda x: x.shape, metrics))

    metrics = net.apply(params, x[:1], method=net.embed_img)
    print(jax.tree.map(lambda x: x.shape, metrics))


    forward_fn = jax.jit(jax.vmap(net.apply, in_axes=(None, 0, 0)))
    bs = 32
    for _ in tqdm(range(50000)):
        x = jnp.zeros((bs, cfg.nt*cfg.pt, cfg.nh*cfg.ph, cfg.nw*cfg.pw), dtype=int)
        loss, _ = forward_fn(params, split(rng, bs), x) # (B, H, W, 2)
        


if __name__ == "__main__":
    main()


