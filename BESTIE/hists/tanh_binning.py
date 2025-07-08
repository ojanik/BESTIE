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

def tanhHistND(
    lss: Array,  # shape (N, D)
    bins_list: Sequence[Array],  # list of D arrays of bin edges
    slopes: Sequence[float],     # list of D floats
):
    """
    N-dimensional differentiable histogram using soft tanh binning.
    Returns:
        mu:  shape (b₁, b₂, ..., b_D)
        ssq: shape (b₁, b₂, ..., b_D)
    """

    #BUG Normalization does not quite work. Optimization still works but the overall scale is off.

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
        combined = combined / jnp.sum(combined)
        return combined  # Normalize to sum to 1

    # Vectorize over all events
    all_weights = vmap(per_event_soft_bin)(lss)  # shape (N, total_bins)

    return all_weights






