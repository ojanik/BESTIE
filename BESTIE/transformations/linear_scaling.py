import jax.numpy as jnp

def linear_scaling(lss,**kwargs):
    try:
        lss = lss#[:,0]
    except:
        pass
    lss -= jnp.min(lss)
    lss /= jnp.max(lss)
    return lss