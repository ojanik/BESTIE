
import jax.numpy as jnp
from jax import vmap
from jax.scipy.stats.norm import cdf as ncdf
import jax.scipy as jsp
Array = jnp.array
from functools import partial
import jax

class bKDE():
    def __init__(self,observables,bandwidth):

        if jnp.ndim(observables) != 1:
            raise Exception("Observable must be 1 dimensional!")

        self.observables = observables
        self.bandwidth = bandwidth

    def cdf(self,t,weights=None):
        if weights is None:
            weights = jnp.ones(len(self.observables))
        cdf = ncdf(t,self.observables,self.bandwidth)
        ccdf = (cdf*weights).sum()/weights.sum()
        return ccdf
    
    def bin_kde(self,bins,weights=None):
        def batch_cdf(t):
            return bKDE.cdf(self,t,weights=weights)
        edges = vmap(batch_cdf)(bins)
        counts = edges[1:]-edges[:-1]
        return counts
    

@partial(jax.jit, static_argnames=["density", "reflect_infinities"])
def hist(
    data: Array,
    bins: Array,
    bandwidth: float,  # | None = None,
    weights: Array,
    density: bool = False,
    reflect_infinities: bool = False,
) -> Array:
    """Differentiable histogram, defined via a binned kernel density estimate (bKDE).

    Parameters
    ----------
    data : Array
        1D array of data to histogram.
    bins : Array
        1D array of bin edges.
    bandwidth : float
        The bandwidth of the kernel. Bigger == lower gradient variance, but more bias.
    density : bool
        Normalise the histogram to unit area.
    reflect_infinities : bool
        If True, define bins at +/- infinity, and reflect their mass into the edge bins.

    Returns
    -------
    Array
        1D array of bKDE counts.
    """
    # bandwidth = bandwidth or events.shape[-1] ** -0.25  # Scott's rule

    bins = jnp.array([-jnp.inf, *bins, jnp.inf]) if reflect_infinities else bins

    # get cumulative counts (area under kde) for each set of bin edges

    cdf = jsp.stats.norm.cdf(bins.reshape(-1, 1), loc=data, scale=bandwidth)
    cdf *= weights
    cdf /= weights.sum()
    # sum kde contributions in each bin
    counts = (cdf[1:, :] - cdf[:-1, :]).sum(axis=1)

    if density:  # normalize by bin width and counts for total area = 1
        db = jnp.array(jnp.diff(bins), float)  # bin spacing
        counts = counts / db / counts.sum(axis=0)

    if reflect_infinities:
        counts = (
            counts[1:-1]
            + jnp.array([counts[0]] + [0] * (len(counts) - 3))
            + jnp.array([0] * (len(counts) - 3) + [counts[-1]])
        )

    return counts
