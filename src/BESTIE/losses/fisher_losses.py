import jax.numpy as jnp

def S_optimality(fisher,signal_idx=[0]):
    cov = -1 * jnp.linalg.inv(fisher)
    uncert = jnp.diag(cov)
    return uncert[signal_idx[0]]

def A_optimality(fisher,signal_idx=None):
    cov = -1 * jnp.linalg.inv(fisher)
    diag = jnp.diag(cov)

    if signal_idx == None:
        trace = jnp.sum(diag)

    else:
        trace = 0
        for idx in signal_idx:
            trace += diag[idx]

    loss = trace

    return loss