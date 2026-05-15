import jax.numpy as jnp
import numpy as onp
Array = jnp.array
from scipy.spatial import KDTree
from functools import partial


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
                                * ``parameters_to_optimize`` (list of names) —
                                  which parameters' per-event FIM diagonal
                                  elements get summed before inversion.
                                  Defaults to ``loss.parameters_to_optimize``
                                  if ``config`` is given. Omitting/empty =
                                  use all parameters.
                                * ``eps`` (float, default 1e-12) — floor on
                                  the info sum to avoid divide-by-zero.
                                * ``chunk_size`` (int, default 1_000_000) —
                                  events processed per inner loop iteration.
        config  (dict, optional): The full top-level config. Used by
                        ``method == "fisher"`` to default
                        ``parameters_to_optimize`` from the ``loss`` section.

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

        # Which parameters' per-event Fisher information are summed before
        # inversion. Defaults to the same parameters the training loss
        # targets (``loss.parameters_to_optimize``) so "use the configured
        # loss" works out of the box; can be overridden under ``sample_weights``.
        parameters_to_optimize = swconfig.get(
            "parameters_to_optimize",
            loss_cfg.get("parameters_to_optimize", None),
        )

        eps = swconfig.get("eps", 1e-12)
        chunk_size = swconfig.get("chunk_size", 1_000_000)

        return partial(
            fisher_sample_weights,
            parameters_to_optimize=parameters_to_optimize,
            eps=eps,
            chunk_size=chunk_size,
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

def fisher_sample_weights(data, weights=None, grad_weights=None,
                          parameters_to_optimize=None,
                          eps=1e-12, chunk_size=1_000_000, **_):
    """Per-event Fisher-information based sampling weights.

    Per-event FIM diagonal for parameter ``k`` is ``g_k^2 / w``. We sum the
    diagonals over the parameters in ``parameters_to_optimize`` to get a
    scalar "info content" per event, then take its inverse as the sampling
    weight (events with smaller selected-parameter info get more sampling
    probability). The aggregate-FIM estimate is kept unbiased by the
    importance-sampling reweight inside ``Dataset.get_sampler``.

    Notes:
        - No matrix inversion, no Schur complement, no optimality function —
          everything is scalar per event, so the routine runs in pure
          numpy float64 on the host and is immune to the float32 NaN issues
          that plagued the loss-based variant.
        - If ``parameters_to_optimize`` is None or empty, all parameters are
          included in the diagonal sum.

    Args:
        data: (N, D) input array. Only used for the length.
        weights: (N,) per-event MC weights. Required.
        grad_weights: dict {param_name: (N,) array of ∂w/∂θ}. Required.
        parameters_to_optimize: list of parameter names whose per-event FIM
            diagonal contributions are summed. Defaults to all parameters.
        eps: floor on |w| and on the info sum to avoid /0.
        chunk_size: process events in chunks of this many to keep peak
            memory modest. The per-chunk peak is roughly ``chunk_size *
            (P_sel + 1) * 8`` bytes — at the default 1M and P_sel ~ a few
            it's well under 100 MB.

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

    # Alphabetical key order matches Dataset/Pipeline / training conventions,
    # though only the names matter here — we never index into a parameter
    # vector by position.
    keys = sorted(grad_weights.keys())
    N = len(weights)

    if parameters_to_optimize:
        selected = [k for k in parameters_to_optimize if k in keys]
        if not selected:
            raise ValueError(
                "None of the requested parameters_to_optimize "
                f"{parameters_to_optimize} are present in grad_weights "
                f"(have: {keys})."
            )
    else:
        selected = list(keys)

    weights_np = onp.asarray(weights, dtype=onp.float64)
    # Take views per parameter rather than stacking — we only need the
    # selected ones, and the full (N, P) stack is unnecessary memory.
    grads_np = {k: onp.asarray(grad_weights[k], dtype=onp.float64)
                for k in selected}

    info = onp.empty(N, dtype=onp.float64)
    for start in range(0, N, chunk_size):
        stop = min(start + chunk_size, N)
        w_safe = onp.maximum(weights_np[start:stop], eps)
        s = onp.zeros(stop - start, dtype=onp.float64)
        for k in selected:
            g_k = grads_np[k][start:stop]
            s += (g_k * g_k) / w_safe
        info[start:stop] = s

    # Sample weight ∝ 1 / per-event info. NaNs (shouldn't happen, but if
    # any grad column has them) are treated as zero info => upweighted; if
    # that's the wrong behaviour for your data, fix the NaNs upstream.
    info = onp.where(onp.isnan(info), 0.0, info)
    inv = 1.0 / (info + eps)
    sample_weights = inv / inv.sum()
    return jnp.asarray(sample_weights)
