import jax.numpy as jnp
from jax import vmap
Array = jnp.array

def tanh_norm(bin_width,slope):
    """Return the normalization factor for a tanh binning."""
    return 1.0 / jnp.tanh(bin_width/slope)


def tanh_binning(x, bin_edges, slope, weight):
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
    bin_width = (jnp.max(bins) - jnp.min(bins))/len(bins)
    norm = tanh_norm(bin_width,slope)
    tanh_binning_vmap = vmap(tanh_binning, in_axes=(0, None, None, 0))
    mu = 1/norm * tanh_binning_vmap(lss, bins, slope, mu_weights)
    ssq = 1/norm * tanh_binning_vmap(lss, bins, slope, ssq_weights)
    mu = jnp.sum(mu, axis=0)
    ssq = jnp.sum(ssq, axis=0)
    return mu, ssq
