import jax.numpy as jnp

def linear_scaling(lss,**kwargs):
    lss -= jnp.min(lss)
    lss /= jnp.max(lss)
    return lss