import jax.numpy as jnp
from jax import vmap
Array = jnp.array
from typing import Sequence
from functools import reduce

import jax

def tanh_norm(bin_width,slope):
    """Return the normalization factor for a tanh binning."""
    return 1.0 / jnp.tanh(bin_width/slope)


def tanh_binning(x, bin_edges, slope, weight=1.):
    """Return the bin index for a tanh binning."""
    return (0.5*(jnp.tanh((x - bin_edges[:-1]) / slope) * jnp.tanh(-(x - bin_edges[1:]) / slope)+1)) * weight

def tanhHist(
    lss: Array,
    bins: Array,
    slope: float,
    mu_weights: Array,
    ssq_weights: Array,):
    """Differentiable histogram, defined via a tanh binning.
    Parameters
    ----------
    data : Array
        1D array of data to histogram.
    bins : Array
        1D array of bin edges.
    slope : float
        The slope of the tanh function.
    Returns
    -------
    Array
        1D array of tanhHist counts.
    """
    #TODO add handling of events lying at the edges of the binning range
    bin_width = (jnp.max(bins) - jnp.min(bins))/len(bins)
    norm = tanh_norm(bin_width,slope)
    tanh_binning_vmap = vmap(tanh_binning, in_axes=(0, None, None, 0))
    mu = 1/norm * tanh_binning_vmap(lss, bins, slope, mu_weights)
    ssq = 1/norm * tanh_binning_vmap(lss, bins, slope, ssq_weights)
    mu = jnp.sum(mu, axis=0)
    ssq = jnp.sum(ssq, axis=0)
    return mu, ssq

def tanhHistND(
    lss: Array,  # shape (N, D)
    bins_list: Sequence[Array],  # list of D arrays of bin edges
    slopes: Sequence[float],     # list of D floats
    weights: Array,           # shape (N,)
):
    """
    N-dimensional differentiable histogram using soft tanh binning.
    Returns:
        mu:  shape (b₁, b₂, ..., b_D)
        ssq: shape (b₁, b₂, ..., b_D)
    """
    N, D = lss.shape
    
    assert D == len(bins_list) == len(slopes), "Mismatch in lss dimensions and binning lists"

    bin_counts = [len(bins) - 1 for bins in bins_list]
    norm_factors = [tanh_norm((jnp.max(b) - jnp.min(b)) / (len(b) - 1), s) for b, s in zip(bins_list, slopes)]

    def per_event_soft_bin(x_event):
        """Returns per-event soft bin memberships across D dims."""
        soft_bins = []
        for d in range(D):
            memberships = tanh_binning(x_event[d], bins_list[d], slopes[d])
            memberships = memberships * norm_factors[d]
            soft_bins.append(memberships)
        # Tensor product (outer product) across all dimensions
        combined = reduce(lambda a, b: jnp.outer(a, b).reshape(-1), soft_bins)
        return combined  # Normalize to sum to 1

    # Vectorize over all events
    all_weights = vmap(per_event_soft_bin)(lss)  # shape (N, total_bins)
    counts = jnp.sum(all_weights * weights[:, None], axis=0)

    return Array(counts)