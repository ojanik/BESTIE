import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Union

class FourierFeatures(nn.Module):
    n_frequencies: int
    sigma: Union[float, Sequence[float]] = 1.0   # scalar or (in_dim,)
    trainable: bool = False

    @nn.compact
    def __call__(self, x):
        # x: (..., in_dim)
        in_dim = x.shape[-1]

        def init_B(key, shape):
            # B ~ N(0, 1), then scale by sigma (scalar or per-dim)
            B = jax.random.normal(key, shape, dtype=x.dtype)  # (F, D)
            s = jnp.asarray(self.sigma, dtype=x.dtype)
            if s.ndim == 0:
                return B * s
            elif s.shape == (in_dim,):
                return B * s  # (F, D) * (D,) -> (F, D)
            else:
                raise ValueError(
                    f"'sigma' must be scalar or shape ({in_dim},), got {s.shape}"
                )

        B = self.param("B", init_B, (self.n_frequencies, in_dim))
        if not self.trainable:
            B = jax.lax.stop_gradient(B)

        z = jnp.einsum("...d,fd->...f", x, B) * (2.0 * jnp.pi)
        return jnp.concatenate([jnp.sin(z), jnp.cos(z)], axis=-1)


class MultiScaleFourierFeatures(nn.Module):
    n_frequencies: int
    sigmas: Sequence[Union[float, Sequence[float]]]  # list of scalars or (in_dim,) vectors
    trainable: bool = False

    @nn.compact
    def __call__(self, x):
        in_dim = x.shape[-1]
        outs = []

        for i, sigma in enumerate(self.sigmas):
            def init_B(key, shape, sigma=sigma):
                B = jax.random.normal(key, shape, dtype=x.dtype)  # (F, D)
                s = jnp.asarray(sigma, dtype=x.dtype)
                if s.ndim == 0:
                    return B * s
                elif s.shape == (in_dim,):
                    return B * s
                else:
                    raise ValueError(
                        f"'sigma[{i}]' must be scalar or shape ({in_dim},), got {s.shape}"
                    )

            B = self.param(f"B_{i}", init_B, (self.n_frequencies, in_dim))
            if not self.trainable:
                B = jax.lax.stop_gradient(B)

            z = jnp.einsum("...d,fd->...f", x, B) * (2.0 * jnp.pi)
            outs.append(jnp.concatenate([jnp.sin(z), jnp.cos(z)], axis=-1))

        return outs  # list of encodings

"""import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

class FourierFeatures(nn.Module):
    n_frequencies: int            # F
    sigma: float = 1.0            # frequency scale (std of Gaussian)
    trainable: bool = False       # usually keep B fixed

    @nn.compact
    def __call__(self, x):
        # x: (..., in_dim)
        in_dim = x.shape[-1]
        # B ~ N(0, sigma^2)
        B = self.param(
            "B",
            lambda key, shape: self.sigma * jax.random.normal(key, shape),
            (self.n_frequencies, in_dim),
        )
        if not self.trainable:
            B = jax.lax.stop_gradient(B)

        # z = x @ B^T -> (..., F)
        z = jnp.einsum("...d,fd->...f", x, B)
        z = 2.0 * jnp.pi * z

        # return (..., 2F)
        return jnp.concatenate([jnp.sin(z), jnp.cos(z)], axis=-1)
    
class MultiScaleFourierFeatures(nn.Module):
    n_frequencies: int
    sigmas: Sequence[float]
    trainable: bool = False  # keep B fixed by default

    @nn.compact
    def __call__(self, x):
        # x: (..., in_dim)
        in_dim = x.shape[-1]
        outs = []
        for i, sigma in enumerate(self.sigmas):
            B = self.param(
                f"B_{i}",
                lambda key, shape, s=sigma: s * jax.random.normal(key, shape),
                (self.n_frequencies, in_dim),
            )
            if not self.trainable:
                B = jax.lax.stop_gradient(B)

            z = jnp.einsum("...d,fd->...f", x, B)  # (..., F)
            z = 2.0 * jnp.pi * z
            enc = jnp.concatenate([jnp.sin(z), jnp.cos(z)], axis=-1)  # (..., 2F)
            outs.append(enc)
        return outs  # list of length len(sigmas); each: (..., 2F)s"""