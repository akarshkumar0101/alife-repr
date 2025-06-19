import jax
import jax.numpy as jnp
import flax.linen as nn

class Block(nn.Module):
    """A block of a CNN."""
    channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1

    @nn.compact
    def __call__(self, x):
        identity = x
        x = nn.LayerNorm()(x)
        # x = nn.Conv(self.channels, self.kernel_size, self.stride, self.padding)(x)
        x = jnp.pad(x, pad_width=[(1, 1), (1, 1), (0, 0)], mode='wrap')
        x = nn.Conv(self.channels, (3, 3), (1, 1), padding='VALID')(x)
        x = nn.gelu(x)

        x = nn.LayerNorm()(x)
        # x = nn.Conv(self.channels, self.kernel_size, self.stride, self.padding)(x)
        x = jnp.pad(x, pad_width=[(1, 1), (1, 1), (0, 0)], mode='wrap')
        x = nn.Conv(self.channels, (3, 3), (1, 1), padding='VALID')(x)
        x = nn.gelu(x)

        x = identity + x
        return x

class ConvNet(nn.Module):
    layers: int
    channels: int
    out_channels: int
    n_dts: int = 1

    @nn.compact
    def __call__(self, x, dt_id: int = 0, return_hidden_reprs: bool = False):
        """
        x: (H, W)
        dt_id: int describing the which dt to use, 0 is the first dt, 1 is the second dt, etc.
        """
        hidden_reprs = []
        x = jax.nn.one_hot(x, num_classes=2) # (H, W, 2)
        x = nn.Conv(self.channels, 1, 1, 0)(x) # (H, W, D)
        dt_id = nn.Embed(self.n_dts, self.channels)(dt_id) # (D, )
        x = x + dt_id # (H, W, D)
        for _ in range(self.layers):
            x = Block(self.channels)(x)
            hidden_reprs.append(x)
        x = nn.Conv(self.out_channels, 1, 1, 0, kernel_init=nn.initializers.normal(0.01))(x)
        return (x, hidden_reprs) if return_hidden_reprs else x
    
    @nn.compact
    def forward_soft(self, x, dt_id: int = 0, return_hidden_reprs: bool = False):
        """
        x: (H, W, 2)
        dt_id: int describing the which dt to use, 0 is the first dt, 1 is the second dt, etc.
        """
        hidden_reprs = []
        x = nn.Conv(self.channels, 1, 1, 0)(x) # (H, W, D)
        dt_id = nn.Embed(self.n_dts, self.channels)(dt_id) # (D, )
        x = x + dt_id # (H, W, D)
        for _ in range(self.layers):
            x = Block(self.channels)(x)
            hidden_reprs.append(x)
        x = nn.Conv(self.out_channels, 1, 1, 0, kernel_init=nn.initializers.normal(0.01))(x)
        return (x, hidden_reprs) if return_hidden_reprs else x

