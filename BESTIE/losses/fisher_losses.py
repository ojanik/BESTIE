import jax.numpy as jnp

def calc_conv(fisher):
    return jnp.linalg.inv(fisher) *-1.

def S_optimality(fisher,signal_idx=[0]):
    cov = calc_conv(fisher)
    uncert = jnp.diag(cov)
    return uncert

def A_optimality(fisher,signal_idx=None):
    cov = calc_conv(fisher)
    diag = jnp.diag(cov)

    trace = jnp.sum(jnp.sqrt(diag))

    """else:
        trace = 0
        for idx in signal_idx:
            trace += jnp.sqrt(diag.at[idx].get(mode='fill', fill_value=jnp.nan))

    """
    loss = trace

    return loss

def Matthias_loss(fisher,signal_idx=None):
    cov = calc_conv(fisher)


    sig_phi = 1.405
    sig_gamma = 0.368

    return cov[0,0]/sig_phi**2 + cov[1,1]/sig_gamma**2 + 2*cov[0,1]/(sig_phi*sig_gamma)

def D_optimality(fisher,signal_idx=None):
    return 1/jnp.sqrt(jnp.linalg.det(fisher))
