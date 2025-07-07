import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from dataclasses import dataclass
from einops import rearrange, repeat
import optax

from .transformer import MLP, CausalSelfAttention, GPTConfig

class TransformerBlock(nn.Module):
    n_embd: int
    n_head: int

    def setup(self):
        self.cfg = GPTConfig(n_embd=self.n_embd, n_head=self.n_head, dropout=0.)
        self.ln_1 = nn.LayerNorm(epsilon=1e-5)
        self.attn = CausalSelfAttention(self.cfg)
        self.ln_2 = nn.LayerNorm(epsilon=1e-5)
        self.mlp = MLP(self.cfg)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        x.shape: (T, D)
        """
        x = rearrange(x, "... -> 1 ...")
        x = x + self.attn(self.ln_1(x), train=True)
        x = x + self.mlp(self.ln_2(x), train=True)
        x = rearrange(x, "1 ... -> ...")
        return x

class ConvBlock(nn.Module):
    n_embd: int
    kernel_size: int = 3

    def setup(self):
        self.cfg = GPTConfig(n_embd=self.n_embd, n_head=0, dropout=0.)
        self.ln_1 = nn.LayerNorm(epsilon=1e-5)
        self.conv = nn.Conv(self.n_embd, (3, 3), (1, 1), padding='VALID')
        self.ln_2 = nn.LayerNorm(epsilon=1e-5)
        self.mlp = MLP(self.cfg)

    def __call__(self, x):
        """
        x.shape: (T, D)
        """
        x = rearrange(x, "(N M) D -> N M D", N=int(np.sqrt(len(x))))

        a = self.ln_1(x)
        a = jnp.pad(a, pad_width=[(1, 1), (1, 1), (0, 0)], mode='wrap')
        x = x + self.conv(a)
        x = x + self.mlp(self.ln_2(x), train=True)
        x = rearrange(x, "N M D -> (N M) D")
        return x

class ConvBlockOld(nn.Module):
    n_embd: int
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x):
        """
        x.shape: (T, D)
        """
        x = rearrange(x, "(N M) D -> N M D", N=int(np.sqrt(len(x))))
        identity = x
        x = nn.LayerNorm()(x)
        x = jnp.pad(x, pad_width=[(1, 1), (1, 1), (0, 0)], mode='wrap')
        x = nn.Conv(self.n_embd, (3, 3), (1, 1), padding='VALID')(x)
        x = nn.gelu(x)

        x = nn.LayerNorm()(x)
        x = jnp.pad(x, pad_width=[(1, 1), (1, 1), (0, 0)], mode='wrap')
        x = nn.Conv(self.n_embd, (3, 3), (1, 1), padding='VALID')(x)
        x = nn.gelu(x)

        x = identity + x
        x = rearrange(x, "N M D -> (N M) D")
        return x
    
@dataclass
class DTNetworkConfig:
    layers: int = 1
    n_embd: int = 16
    n_head: int = 4 # only used for transformer block
    block: str = "attn"

    nt: int = 1 # number of patches in time dimension
    nh: int = 16 # number of patches in height dimension
    nw: int = 16 # number of patches in width dimension
    pt: int = 1 # patch size in time dimension
    ph: int = 1 # patch size in height dimension
    pw: int = 1 # patch size in width dimension


class DTNetwork(nn.Module):
    cfg: DTNetworkConfig

    def setup(self):
        cfg = self.cfg
        if self.cfg.block == "old_conv":
            self.blocks = [ConvBlockOld(n_embd=self.cfg.n_embd) for _ in range(self.cfg.layers)]
        elif self.cfg.block == "conv":
            self.blocks = [ConvBlock(n_embd=self.cfg.n_embd) for _ in range(self.cfg.layers)]
        elif self.cfg.block == "attn":
            self.blocks = [TransformerBlock(n_embd=self.cfg.n_embd, n_head=self.cfg.n_head) for _ in range(self.cfg.layers)]
            self.pos_embed = nn.Embed(cfg.nt*cfg.nh*cfg.nw, cfg.n_embd)
        else:
            raise ValueError(f"Invalid block type: {self.cfg.block}")
        cfg = self.cfg
        self.patch_embed = nn.Dense(cfg.n_embd)
        self.head = nn.Dense(cfg.pt*cfg.ph*cfg.pw)

    def __call__(self, x, y):
        """
        x: (H, W)
        """
        x = rearrange(x, "H W -> 1 H W")
        cfg = self.cfg
        x_in = x
        x = x.astype(float)
        # convert to flattened patches
        x = rearrange(x, "(nt pt) (nh ph) (nw pw) -> (nt nh nw) (pt ph pw)",
                      nt=cfg.nt, nh=cfg.nh, nw=cfg.nw, pt=cfg.pt, ph=cfg.ph, pw=cfg.pw)
        # embed patches
        x = self.patch_embed(x)
        # add positional embeddings
        if self.cfg.block == "transformer":
            x = x + self.pos_embed(jnp.arange(len(x)))
        # forward encoder
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        features = jnp.stack(features)
        # project to patches
        x = self.head(x)
        # unflatten patches
        x = rearrange(x, "(nt nh nw) (pt ph pw) -> (nt pt) (nh ph) (nw pw)",
                      nt=cfg.nt, nh=cfg.nh, nw=cfg.nw, pt=cfg.pt, ph=cfg.ph, pw=cfg.pw)
        logits = rearrange(x, "1 H W -> H W")
        probs = jax.nn.sigmoid(logits)
        loss_ce = optax.losses.sigmoid_binary_cross_entropy(logits, y)
        loss = loss_ce.mean()
        metrics = dict(x=x_in, y=y, features=features, logits=logits, probs=probs, loss_ce=loss_ce, loss=loss)
        return loss, metrics