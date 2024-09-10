import jax.numpy as jnp

def vector_hist_old(lss,weights):
    lss = lss *weights
    counts = jnp.sum(lss,axis=0)

    return counts

def vector_hist(lss,weights):
    counts = jnp.dot(jnp.transpose(weights),lss)
    
    return jnp.squeeze(counts)