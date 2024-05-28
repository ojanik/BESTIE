import jax.numpy as jnp

def A_optimality(fisher,signal_idx=None):
    cov = jnp.linalg.inv(fisher)
    diag = jnp.diag(cov)
    

    if signal_idx == None:
        trace = jnp.sum(diag)

    else:
        trace = 0
        for idx in signal_idx:
            trace += diag[idx]

    loss = trace

    return loss