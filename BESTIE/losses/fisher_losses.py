import jax.numpy as jnp

from jax import grad, jacfwd, vmap
import jax
import jax.numpy as jnp 
from BESTIE.utilities import rearrange_matrix

from jax import hessian

from functools import partial

def loss_fisher_jac(llh, injected_params, lss, aux, data_hist, sample_weights, **kwargs):
    signal_idx = kwargs.pop("signal_idx")
    opti = kwargs.pop("opti")
    weight_norm = kwargs.pop("weight_norm", None)

    # Optional soft masking hyperparameters
    threshold = kwargs.pop("rel_uncertainty_threshold", 0.05)  # e.g. 20% relative uncertainty
    sharpness = kwargs.pop("mask_sharpness", 200)           # how steep the sigmoid is
    eps = 1e-8

    # Get gradients and MC uncertainty terms
    grads, sigma = jacfwd(llh, has_aux=True)(
        injected_params, lss, aux, data_hist, sample_weights,
        skip_llh=True, **kwargs
    )
    mu, sigma = llh(
        injected_params, lss, aux, data_hist, sample_weights,
        skip_llh=True, **kwargs
    )

    

    # Compute per-bin Fisher information
    fish_i = vmap(lambda g: jnp.outer(g, g))(grads)  # shape: (n_bins, n_params, n_params)
    hist_counts = mu + eps
    fish_i = fish_i / hist_counts[:, None, None]

    if threshold is not None or sharpness is not None:
        # Soft mask: 1.0 for good bins, ~0.0 for noisy ones
        rel_unc = jnp.sqrt(sigma + eps) / (mu + eps)
        soft_mask = 1.0 - jax.nn.sigmoid((rel_unc - threshold) * sharpness)  # shape: (n_bins,)
        # Apply soft mask (broadcasted to matrix shape)
        fish_i = fish_i * soft_mask[:, None, None]

    # Sum over bins
    fish = jnp.sum(fish_i, axis=0)  # shape: (n_params, n_params)

    # Marginalize Fisher info
    fish = rearrange_matrix(fish, signal_idx)
    k = len(signal_idx)
    A = fish[:k, :k]
    B = fish[:k, k:]
    C = fish[k:, k:]
    C_inv = jnp.linalg.inv(C)
    S = A - B @ C_inv @ B.T

    return opti(S, weight_norm)

def calc_conv(fisher):
    return jnp.linalg.inv(fisher) #*-1.

def A_optimality(fisher,weight_norm=None):
    cov = calc_conv(fisher)
    diag = jnp.diag(cov)

    if weight_norm is not None:
        if isinstance(weight_norm,list):
            weight_norm = jnp.array(weight_norm)
        trace = jnp.sum(jnp.sqrt(diag)/weight_norm)
    else:
        trace = jnp.sum(jnp.sqrt(diag))
    loss = trace

    return loss

def D_optimality(fisher,signal_idx=None):
    return 1/jnp.sqrt(jnp.linalg.det(fisher))
