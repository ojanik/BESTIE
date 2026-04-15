from typing import Any, List
import jax
import jax.numpy as jnp
from flax import linen as nn

class FourierFeatures(nn.Module):
    n_frequencies: int
    sigmas: Any  # list of scales: each element scalar or length-D list
    trainable: bool = False
    concat_raw_input: bool = False

    @nn.compact
    def __call__(self, x):
        # x: (..., D)
        D = x.shape[-1]
        F = int(self.n_frequencies)

        if not isinstance(self.sigmas, (list, tuple)):
            raise TypeError(
                f"sigma must be a list of scales; each scale is scalar or length-D list. Got {type(self.sigmas)}"
            )
        if len(self.sigmas) == 0:
            raise ValueError("sigma must be a non-empty list of scales.")

        # Build sigma_sd with shape (S, D)
        rows: List[jnp.ndarray] = []
        for i, sc in enumerate(self.sigmas):
            if isinstance(sc, (int, float)):
                rows.append(jnp.full((D,), float(sc), dtype=x.dtype))
            elif isinstance(sc, (list, tuple)):
                if len(sc) != D:
                    raise ValueError(
                        f"sigma[{i}] must be scalar or length D={D}. Got length {len(sc)}."
                    )
                rows.append(jnp.asarray(sc, dtype=x.dtype))
            else:
                raise TypeError(f"sigma[{i}] has invalid type {type(sc)}; must be scalar or list/tuple.")

        sigma_sd = jnp.stack(rows, axis=0)  # (S, D)
        S = int(sigma_sd.shape[0])

        def init_B(key, shape):
            # shape: (S, F, D)
            B0 = jax.random.normal(key, shape, dtype=x.dtype)
            return B0 * sigma_sd[:, None, :]  # (S, F, D)

        B = self.param("B", init_B, (S, F, D))
        if not self.trainable:
            B = jax.lax.stop_gradient(B)

        # (..., S, F)
        z = jnp.einsum("...d,sfd->...sf", x, B) * (2.0 * jnp.pi)

        # (..., S, 2F) -> (..., S*2F)
        y = jnp.concatenate([jnp.sin(z), jnp.cos(z)], axis=-1)
        y = y.reshape(*x.shape[:-1], S * (2 * F))

        if self.concat_raw_input:
            y = jnp.concatenate([x, y], axis=-1)

        return y