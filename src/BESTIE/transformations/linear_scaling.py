import jax.numpy as jnp

def linear_scaling(lss,**kwargs):
    lss = lss[:,0]
    lss -= jnp.min(lss)
    lss /= jnp.max(lss)
    return lss