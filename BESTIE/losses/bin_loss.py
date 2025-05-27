import jax.numpy as jnp


def bin_loss(llh, injected_params, lss, aux, data_hist, sample_weights, **kwargs):
    mu, sigma = llh(
        injected_params, lss, aux, data_hist, sample_weights,
        skip_llh=True, **kwargs
    )
    
    ds_length = kwargs.pop("df_length", None)
    threshold = kwargs.pop("threshold", 0.1)  # Default 10% rel MC error if not given

    eps_mu = 1e-6
    eps_sigma = 1e-12

    # Relative MC uncertainty: sqrt(ssq) / mu
    rel_unc = jnp.sqrt(sigma + eps_sigma) / (mu + eps_mu)

    # Optional: mask out empty bins (mu == 0)
    mask = mu > 0
    rel_unc_masked = jnp.where(mask, rel_unc, 0.0)

    # Penalize only if rel_unc exceeds threshold
    penalty = jnp.mean(jnp.where(mask, jnp.maximum(rel_unc_masked - threshold, 0.0)**2, 0.0))

    return penalty
