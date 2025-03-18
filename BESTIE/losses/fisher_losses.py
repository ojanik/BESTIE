import jax.numpy as jnp

from jax import grad, jacfwd, vmap
import jax
import jax.numpy as jnp 
from BESTIE.utilities import rearrange_matrix


def loss_fisher_jac(llh,injected_params,lss,aux,data_hist,sample_weights,**kwargs):

    signal_idx = kwargs.pop("signal_idx")
    opti = kwargs.pop("opti")
    weight_norm = kwargs.pop("weight_norm",None)

    grads, sigma = jacfwd(llh,has_aux=True)(injected_params,lss,aux,data_hist,sample_weights,skip_llh=True,**kwargs)
    mu, sigma = llh(injected_params,lss,aux,data_hist,sample_weights,skip_llh=True,**kwargs)
    fish_i = vmap(lambda g: jnp.outer(g, g))(grads)
    hist_counts = mu + 1e-8#llh(injected_params,lss,aux,data_hist,sample_weights,skip_llh=True,**kwargs) + 1e-8
    nonzero_hist_counts = hist_counts != 0
    fish_i = jnp.where(nonzero_hist_counts[:, None, None],
                        fish_i / hist_counts[:, None, None], 
                        jnp.zeros(fish_i.shape))  # Shape: (1650, 11, 11)
    fish = jnp.sum(fish_i, axis=0) 

    # marginalize fisher information
    fish = rearrange_matrix(fish, signal_idx)
    k = len(signal_idx)
    A = fish[:k, :k]
    B = fish[:k, k:]
    C = fish[k:, k:]

    # Compute the inverse of C
    C_inv = jnp.linalg.inv(C)
    
    # Compute the Schur complement S = A - B * C_inv * B^T
    S = A - B @ C_inv @ B.T

    return opti(S,weight_norm)


def loss_fisher(llh,injected_params,lss,aux,data_hist,sample_weights,**kwargs):
    #grads = grad(llh)(injected_params,lss,aux,data_hist,sample_weights,**kwargs)
    #fish = -1 * jnp.outer(grads,grads)
    #jax.debug.print("fish grads: {fish}",fish=fish)
    fish = hessian(llh)(injected_params,lss,aux,data_hist,sample_weights,**kwargs)

    # marginalize fisher information
    fish = rearrange_matrix(fish, lconfig["signal_idx"])
    k = len(lconfig["signal_idx"])
    A = fish[:k, :k]
    B = fish[:k, k:]
    C = fish[k:, k:]

    # Compute the inverse of C
    C_inv = onp.linalg.inv(C)

    # Compute the Schur complement S = A - B * C_inv * B^T
    S = A - B @ C_inv @ B.T

    return opti(S)

def calc_conv(fisher):
    return jnp.linalg.inv(fisher) #*-1.

def S_optimality(fisher,signal_idx=[0]):
    cov = calc_conv(fisher)
    uncert = jnp.diag(cov)
    return uncert

def A_optimality(fisher,weight_norm=None):
    cov = calc_conv(fisher)
    diag = jnp.diag(cov)

    if weight_norm is not None:
        if isinstance(weight_norm,list):
            weight_norm = jnp.array(weight_norm)
        trace = jnp.sum(jnp.sqrt(diag)/weight_norm)
    else:
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
