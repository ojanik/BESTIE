import jax.numpy as jnp

def vector_hist(lss,weights):
    lss = lss *weights
    counts = jnp.sum(lss,axis=0)

    return counts