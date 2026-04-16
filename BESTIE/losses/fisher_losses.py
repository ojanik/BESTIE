import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten

from ..utilities import rearrange_matrix



def fisher_loss(mu,ssq,grad_hist,**kwargs):
    #BUG overall scale of loss calculated here is off. Minimization still works.
    parameters_to_optimize = kwargs.pop("parameters_to_optimize")
    opti = kwargs.pop("opti")
    weight_norm = kwargs.pop("weight_norm", None)
    fim_reg = kwargs.pop("fim_regularization", 1e-3)

    # Optional soft masking hyperparameters
    threshold = kwargs.pop("rel_uncertainty_threshold", 0.2)  # e.g. 20% relative uncertainty
    sharpness = kwargs.pop("mask_sharpness", 200)           # how steep the sigmoid is
    softmasking = kwargs.pop("use_softmasking",True)
    eps = 1e-8

    information = jax.tree_util.tree_map(lambda v: v/(jnp.sqrt(mu+1e-8)), grad_hist)
    
    
    flat_values, _ = tree_flatten(information)
    values = jnp.stack(flat_values)
    keys = list(information.keys())

    if softmasking:
        # Soft mask: 1.0 for good bins, ~0.0 for noisy ones
        rel_unc = jnp.sqrt(ssq + eps**2) / (mu + eps)
        soft_mask = 1.0 - jax.nn.sigmoid((rel_unc - threshold) * sharpness)
        # Fold sqrt(mask) into values so the outer product applies the mask once:
        # einsum('ib,jb->ij') gives sum_b v_i(b)*v_j(b), and with sqrt(mask) folded
        # in we get sum_b v_i(b) * mask(b) * v_j(b) as desired.
        values = values * jnp.sqrt(soft_mask)[None, :]

    # Outer product over parameters, summed over bins — avoids materialising
    # the full (n_params, n_params, n_bins) intermediate tensor.
    fisher_information = jnp.einsum('ib,jb->ij', values, values)

    signal_idx = [keys.index(p) for p in parameters_to_optimize]
    fish = rearrange_matrix(fisher_information, signal_idx)

    # Regularise the full FIM before block decomposition. This guarantees the
    # Schur complement S = A - B C^{-1} B^T is positive definite (eigenvalues
    # >= fim_reg), preventing NaN in opti() when the FIM is near-zero early
    # in training. fim_reg should be large enough to stabilise float32 but
    # small relative to the converged FIM values (default 1e-3).
    fish = fish + fim_reg * jnp.eye(fish.shape[0])

    k = len(signal_idx)
    A = fish[:k, :k]
    B = fish[:k, k:]
    C = fish[k:, k:]
    S = A - B @ jnp.linalg.solve(C, B.T)
    return opti(S, weight_norm)

def calc_cov(fisher, reg=1e-3):
    fisher_reg = fisher + reg * jnp.eye(fisher.shape[0])
    return jnp.linalg.inv(fisher_reg)

def A_optimality(fisher,weight_norm=None):
    cov = calc_cov(fisher)
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

def C_optimality(fisher):
    """
    Penalize correlations between parameters.
    Minimizes the squared Frobenius norm of the off-diagonal correlation matrix.
    """
    cov = calc_cov(fisher)
    diag_sqrt = jnp.sqrt(jnp.diag(cov))
    norm = jnp.outer(diag_sqrt, diag_sqrt)
    corr = cov / norm
    off_diag = corr - jnp.diag(jnp.diag(corr))
    return jnp.sum(off_diag ** 2)
