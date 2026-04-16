from flax import linen as nn
import jax.numpy as jnp

class ResNetDenseBlock(nn.Module):
    c_out: int
    act_fn: callable = nn.silu
    dropout: float = 0.0
    use_post_add_activation: bool = False  # usually False for modern PreNorm residuals

    @nn.compact
    def __call__(self, x, *, training: bool = False):
        c_in = x.shape[-1]
        skip = x

        # PreNorm residual branch
        h = nn.LayerNorm(epsilon=1e-6)(x)
        h = nn.Dense(self.c_out)(h)
        h = self.act_fn(h)
        h = nn.Dense(self.c_out)(h)

        if self.dropout > 0:
            h = nn.Dropout(rate=self.dropout)(h, deterministic=not training)

        # Project skip if feature dims differ
        if c_in != self.c_out:
            skip = nn.Dense(self.c_out, name="skip_proj")(skip)

        y = skip + h
        if self.use_post_add_activation:
            y = self.act_fn(y)

        return y
class GatedDenseBlock(nn.Module):
    c_out: int
    expansion: int = 4          # d_ff = expansion * c_out
    dropout: float = 0.0
    act_gate: callable = nn.silu # SwiGLU gate; use nn.relu for ReGLU, or identity for GLU

    @nn.compact
    def __call__(self, x, *, training: bool = False):
        c_in = x.shape[-1]
        skip = x

        h = nn.LayerNorm(epsilon=1e-6)(x)

        d_ff = self.expansion * self.c_out

        # Two projections: one for "values", one for "gate"
        v = nn.Dense(d_ff, name="v_proj")(h)
        g = nn.Dense(d_ff, name="g_proj")(h)

        # Gate
        h = v * self.act_gate(g)    # SwiGLU when act_gate = silu

        # Back to width (often zero-init for stable residual start)
        h = nn.Dense(self.c_out, kernel_init=nn.initializers.zeros, name="out_proj")(h)

        if self.dropout > 0:
            h = nn.Dropout(rate=self.dropout)(h, deterministic=not training)

        if c_in != self.c_out:
            skip = nn.Dense(self.c_out, name="skip_proj")(skip)

        return skip + h

class SplitGatedDenseBlock(nn.Module):
    c_out: int                  # must be even if you split evenly
    dropout: float = 0.0
    act_fn: callable = nn.silu

    @nn.compact
    def __call__(self, x, *, training: bool = False):
        assert self.c_out % 2 == 0, "c_out must be even for split gating"
        c_in = x.shape[-1]
        skip = x

        h = nn.LayerNorm(epsilon=1e-6)(x)

        h = nn.Dense(self.c_out)(h)
        h = self.act_fn(h)

        u, v = jnp.split(h, 2, axis=-1)  # each is c_out/2

        # Turn v into a gate (optionally LN + Dense)
        v = nn.LayerNorm(epsilon=1e-6)(v)
        v = nn.Dense(self.c_out // 2)(v)
        gate = nn.sigmoid(v)            # sigmoid gate is common/stable

        h = u * gate

        # Bring back to c_out
        h = nn.Dense(self.c_out, kernel_init=nn.initializers.zeros)(h)

        if self.dropout > 0:
            h = nn.Dropout(rate=self.dropout)(h, deterministic=not training)

        if c_in != self.c_out:
            skip = nn.Dense(self.c_out)(skip)

        return skip + h


def sin(x):
    return jnp.sin(x)

def cos(x):
    return jnp.cos(x)

class sincos(nn.Module):
    

    def setup(self):
        self.alpha = self.param('alpha', nn.initializers.normal(stddev=1),
                                 (1,))

    def __call__(self, X):
        return jnp.sqrt(self.alpha) * jnp.sin(X) + jnp.sqrt(1-self.alpha) * jnp.cos(X)

def sawtooth(x):
    return x%1

def relu(x):
    return nn.relu(x)

def elu(x):
    return nn.elu(x)

def gelu(x):
    return nn.gelu(x)

def silu(x):
    return nn.silu(x)

def softmax(x):
    return nn.softmax(x)

def sigmoid(x):
    return nn.sigmoid(x)

def lin(x):
    return x

def hard_tanh(x):
    return nn.hard_tanh(x)

def tanh(x):
    return jnp.tanh(x)

