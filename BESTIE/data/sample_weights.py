import jax
import jax.numpy as jnp
import numpy as onp
Array = jnp.array
from scipy.spatial import KDTree
from functools import partial

from ..utilities import rearrange_matrix


def sample_weight_handler(dconfig, config=None):
    """
    Returns a sample weighting function based on the configuration.

    Args:
        dconfig (dict): The histogram-level configuration dictionary, expected
                        to have the key ``sample_weights`` containing at least
                        a ``method`` field. Method-specific extra fields:
                            - "hist", "histogram", "binned": ``number_of_sample_bins``
                            - "knn": ``k``
                            - "fisher", "per_event_fisher":
                                * ``optimality`` (e.g. "a"), defaults to the
                                  loss-config optimality if ``config`` is given
                                * ``parameters_to_optimize`` (list of names),
                                  defaults to the loss-config list
                                * ``fim_regularization`` (float, default 1e-3)
                                * ``eps`` (float, default 1e-12) — floor on
                                  1/loss to avoid divide-by-zero
                                * ``weight_norm`` / ``alpha`` / ``beta`` —
                                  forwarded to the optimality function
        config  (dict, optional): The full top-level config. Required when
                        ``method == "fisher"`` so the optimality and
                        signal-parameter list can be picked up from the
                        ``loss`` section as defaults.

    Returns:
        Callable: A function ``calc_sample_weights(data, weights=None,
        grad_weights=None)`` that returns the per-event sampling probability
        (normalized to sum to 1). All methods accept the same signature;
        methods other than "fisher" ignore ``weights`` / ``grad_weights``.

    Raises:
        NotImplementedError: If an unsupported method is specified.
    """
    swconfig = dconfig["sample_weights"]
    method = swconfig["method"].lower()

    if method in {"uniform", "none"}:
        return uniform_sample_weights

    elif method in {"hist", "histogram", "binned"}:
        number_of_sample_bins = swconfig.get("number_of_sample_bins", 50)
        return partial(hist_sample_weights,
                       number_of_sample_bins=number_of_sample_bins)

    elif method in {"knn"}:
        k = swconfig.get("k", 16)
        return partial(knn_sample_weights, k=k)

    elif method in {"fisher", "per_event_fisher"}:
        loss_cfg = (config or {}).get("loss", {}) if config is not None else {}

        # Optimality + signal-parameter defaults come from the loss config so
        # that "use the configured loss" works out of the box; both can be
        # overridden under ``sample_weights`` if you want the proposal to
        # optimise something different from the training loss.
        optimality = swconfig.get("optimality", loss_cfg.get("optimality", "a"))
        parameters_to_optimize = swconfig.get(
            "parameters_to_optimize",
            loss_cfg.get("parameters_to_optimize", None),
        )

        fim_reg = swconfig.get("fim_regularization",
                               loss_cfg.get("fim_regularization", 1e-3))
        eps = swconfig.get("eps", 1e-12)

        # Forward kwargs for the optimality function. weight_norm defaults to
        # the loss-config value; alpha/beta only matter for M-optimality.
        opti_kwargs = {
            "weight_norm": swconfig.get("weight_norm",
                                        loss_cfg.get("weight_norm", None)),
        }
        if "alpha" in swconfig or "alpha" in loss_cfg:
            opti_kwargs["alpha"] = swconfig.get("alpha", loss_cfg.get("alpha"))
        if "beta" in swconfig or "beta" in loss_cfg:
            opti_kwargs["beta"] = swconfig.get("beta", loss_cfg.get("beta"))

        return partial(
            fisher_sample_weights,
            optimality=optimality,
            parameters_to_optimize=parameters_to_optimize,
            fim_reg=fim_reg,
            eps=eps,
            opti_kwargs=opti_kwargs,
        )

    else:
        raise NotImplementedError(f"'{method}' is not a valid sample weight method.")


def uniform_sample_weights(data, weights=None, grad_weights=None, **_):
    """
    Assigns uniform weights to all samples in the dataset.

    Args:
        data (jax.numpy.ndarray): An N x D array of data points.
        weights, grad_weights: unused; kept for a uniform call signature.

    Returns:
        jax.numpy.ndarray: A 1D array of shape (N,) with uniform sample
        weights summing to 1.
    """
    sample_weights = jnp.ones(len(data))
    sample_weights /= jnp.sum(sample_weights)  # Normalize to sum to 1
    return sample_weights


def hist_sample_weights(data, weights=None, grad_weights=None,
                        number_of_sample_bins=20, **_):
    """
    Assigns weights to data points based on their bin occupancy in a
    multi-dimensional histogram. Points in densely populated bins get lower
    weights, and vice versa.

    Args:
        data (jax.numpy.ndarray): An N x D array of data points.
        weights, grad_weights: unused; kept for a uniform call signature.

    Returns:
        jax.numpy.ndarray: A 1D array of shape (N,) with histogram-based
        sample weights.
    """
    digi = []          # List to hold digitized (binned) coordinates for each dimension
    bins_arr = []      # List to store bin edges for each dimension

    for i in range(data.shape[1]):  # Iterate over dimensions
        var = Array(data[:, i])  # Convert column to JAX array
        bins = jnp.linspace(jnp.min(var), jnp.max(var), number_of_sample_bins)  # Bin edges
        bins_arr.append(bins)
        digitized = jnp.digitize(var, bins, right=False) - 1  # Assign bins, start from 0
        digitized = jnp.clip(digitized, 0, number_of_sample_bins - 2)  # Keep within bounds
        digi.append(digitized)

    # Stack digitized results into a 2D array of shape (D, N)
    digi = Array(digi)

    # Compute flattened 1D bin indices for all points
    bin_indices = jnp.ravel_multi_index(digi, [number_of_sample_bins - 1] * data.shape[1])

    # Count how many samples fall into each bin
    counts = jnp.bincount(bin_indices)

    # Assign weight inversely proportional to bin count
    sample_weights = 1 / counts[bin_indices]
    sample_weights /= jnp.sum(sample_weights)  # Normalize to sum to 1
    return sample_weights


def knn_sample_weights(data, weights=None, grad_weights=None, k=16, **_):
    """
    Computes sample weights using k-nearest-neighbors (kNN) density
    estimation. Points in dense regions receive lower weights, and vice versa.

    Args:
        data (jax.numpy.ndarray): An N x D array of data points.
        weights, grad_weights: unused; kept for a uniform call signature.
        k (int): Number of neighbors to consider for density estimation
            (default is 16).

    Returns:
        jax.numpy.ndarray: A 1D array of shape (N,) with kNN-based sample weights.
    """
    print("Calculating kNN weights...")

    data_np = jnp.array(data)  # Convert to NumPy array for KDTree
    tree = KDTree(data_np)     # Build KDTree for fast neighbor queries

    # Get distances to the k+1 nearest neighbors (includes the point itself)
    dists, _ = tree.query(data_np, k=k+1)

    # Sum distances to approximate local density (excluding self)
    knn_density = jnp.sum(dists, axis=1)

    # Use inverse density as sample weights (scaled by data dimensionality)
    sample_weights = knn_density ** len(data[0])
    sample_weights /= jnp.sum(sample_weights)  # Normalize to sum to 1
    return onp.array(sample_weights)


# --------------------------------------------------------------------------- #
# Fisher-information-based proposal
# --------------------------------------------------------------------------- #

def _resolve_opti(optimality):
    """Map the optimality config string to the function in fisher_losses.

    Mirrors the lookup in losses.loss_handler / training.train._get_opti_fn so
    the proposal uses the same scalar criterion as the training loss."""
    o = optimality.lower()
    if o in {"a", "a_optimality", "aoptimality"}:
        from ..losses.fisher_losses import A_optimality
        return A_optimality
    if o in {"c", "c_optimality", "coptimality", "correlation"}:
        from ..losses.fisher_losses import C_optimality
        return C_optimality
    if o in {"d", "d_optimality", "doptimality", "ellipsoid",
             "uncertainty_ellipsoid", "ellipsoid_volume",
             "uncertainty_ellipsoid_volume"}:
        from ..losses.fisher_losses import D_optimality
        return D_optimality
    if o in {"m", "m_optimality", "moptimality", "ac"}:
        from ..losses.fisher_losses import M_optimality
        return M_optimality
    raise NotImplementedError(f"Optimality '{optimality}' not implemented")


def fisher_sample_weights(data, weights=None, grad_weights=None,
                          optimality="a", parameters_to_optimize=None,
                          fim_reg=1e-3, eps=1e-12, opti_kwargs=None, **_):
    """Per-event Fisher-information based sampling weights.

    For each event ``i`` we build a per-event Fisher information matrix from
    its score contribution

        I_i = (g_i g_i^T) / w_i + fim_reg * I,

    where ``g_i`` is the per-parameter gradient vector (``grad_weights``) and
    ``w_i`` the event weight. ``g_i g_i^T`` is rank-1 by construction (outer
    product of a single vector), so the ``fim_reg * I`` regulariser is
    required to make ``I_i`` invertible for criteria like A-/D-optimality —
    same trick used by ``fisher_loss``.

    If ``parameters_to_optimize`` is supplied and is a strict subset of the
    parameter keys, the Schur complement is taken on each per-event FIM so
    nuisance parameters are marginalised before the optimality is evaluated.
    This matches the training-time convention in ``fisher_loss``.

    Sample weight is then ``1 / loss_i`` (normalised to sum to 1): events
    that are individually most informative under the configured criterion
    receive higher sampling probability. The downstream importance-sampling
    correction (``sample_reweights`` in ``Dataset.get_sampler``) keeps the
    aggregated FIM estimate unbiased.

    Caveats:
        - Per-event optimality is a per-event *heuristic*. The criterion on
          the aggregated FIM is non-additive — an individually informative
          event can still be redundant if other events already cover the
          same parameter direction. For variance-optimal proposals see the
          sensitivity formulation (``trace(I_total^-1 I_i)`` and friends).
        - Effective sample size ``(sum w)^2 / sum w^2`` should be tracked
          downstream; an aggressive proposal can shrink it below uniform.

    Args:
        data: (N, D) input array. Used only for the length; the per-event
            score quantities live in ``grad_weights``.
        weights: (N,) per-event MC weights. Required.
        grad_weights: dict {param_name: (N,) array of ∂w/∂θ}. Required.
        optimality: optimality string, same vocabulary as
            ``loss.fisher_losses`` (a / c / d / m).
        parameters_to_optimize: optional list of signal-parameter names; if
            provided and shorter than ``len(grad_weights)``, Schur complement
            is applied per event before evaluating the optimality.
        fim_reg: regulariser added to each per-event FIM.
        eps: floor on the per-event loss when inverting (avoids /0 if a
            poorly conditioned event slips through).
        opti_kwargs: dict of extra kwargs forwarded to the optimality
            function (weight_norm, alpha, beta).

    Returns:
        jax.numpy.ndarray: (N,) sample weights, summing to 1.
    """
    if weights is None or grad_weights is None:
        raise ValueError(
            "fisher_sample_weights requires both `weights` and `grad_weights`."
        )
    if len(grad_weights) == 0:
        raise ValueError(
            "fisher_sample_weights needs at least one parameter in grad_weights."
        )

    opti_fn = _resolve_opti(optimality)
    opti_kwargs = dict(opti_kwargs or {})

    # Match the alphabetical sort used in Dataset/Pipeline, so signal/nuisance
    # indices line up with the training-time convention.
    keys = sorted(grad_weights.keys())
    P = len(keys)

    # Per-event score matrix G: shape (N, P)
    G = jnp.stack([jnp.asarray(grad_weights[k]) for k in keys], axis=1)
    w = jnp.maximum(jnp.asarray(weights), eps)

    # I_i = g_i g_i^T / w_i + fim_reg * I    -> shape (N, P, P)
    # Spreading 1/sqrt(w) into G keeps the einsum/vmap symmetric and matches
    # the unbinned-FIM convention used in train._compute_standard_hist_baseline
    # (fim = (G/w[:,None]).T @ G, i.e. sum_i g_i g_i^T / w_i).
    G_scaled = G / jnp.sqrt(w)[:, None]
    I_per = jnp.einsum('ni,nj->nij', G_scaled, G_scaled)
    I_per = I_per + fim_reg * jnp.eye(P)

    # Optional Schur complement per event, mirroring fisher_loss. We compute
    # signal_idx here (Python-side, static) so the per-event function below
    # vmaps cleanly.
    if parameters_to_optimize is not None:
        signal_idx = [keys.index(p) for p in parameters_to_optimize if p in keys]
    else:
        signal_idx = list(range(P))
    k_sig = len(signal_idx)

    if 0 < k_sig < P:
        def schur_one(fim):
            fim_r = rearrange_matrix(fim, signal_idx)
            A = fim_r[:k_sig, :k_sig]
            B = fim_r[:k_sig, k_sig:]
            C = fim_r[k_sig:, k_sig:]
            return A - B @ jnp.linalg.inv(C) @ B.T
        I_target = jax.vmap(schur_one)(I_per)
    else:
        # All parameters are signal (or none) -> no marginalisation.
        I_target = I_per

    # Per-event optimality loss. opti functions expect a single (k,k) matrix
    # and return a scalar; vmap over the leading event axis.
    def loss_one(fim):
        return opti_fn(fim, **opti_kwargs)
    losses = jax.vmap(loss_one)(I_target)

    # Sample weight ∝ 1 / per-event loss. Floor with eps to be safe even
    # though fim_reg already keeps the FIM well-conditioned.
    inv_loss = 1.0 / (losses + eps)
    sample_weights = inv_loss / jnp.sum(inv_loss)
    return sample_weights
