
import jax.numpy as jnp
from jax.scipy.stats.norm import cdf as ncdf
import jax.scipy as jsp
Array = jnp.array
from functools import partial
import jax

@partial(jax.jit, static_argnames=["density", "reflect_infinities"])
def bKDE(
    lss: Array,
    bins: Array,
    bandwidth: float,  # | None = None,
    mu_weights: Array,
    ssq_weights: Array,
    density: bool = False,
    reflect_infinities: bool = True,
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
    #bw = jax.nn.sigmoid(lss[:,1])
    #lss = lss[:,0]
    #lss -= jnp.min(lss)
    #lss /= jnp.max(lss)

    bins = jnp.array([-jnp.inf, *bins, jnp.inf]) if reflect_infinities else bins

    # get cumulative counts (area under kde) for each set of bin edges

    cdf = jsp.stats.norm.cdf(bins.reshape(-1, 1), loc=lss, scale=bandwidth)
    mu_weights = mu_weights.squeeze()
    mu_cdf = cdf * mu_weights
    #cdf /= weights.sum()
    # sum kde contributions in each bin
    counts = (mu_cdf[1:, :] - mu_cdf[:-1, :]).sum(axis=1)

    calc_sigma = True # TODO add implementation of SAY likelihood, then activate this

    if calc_sigma:
        ssq_weights = ssq_weights.squeeze()
        ssq_cdf = cdf * ssq_weights
        sigma = ((ssq_cdf[1:, :] - ssq_cdf[:-1, :])).sum(axis=1)

    if density:  # normalize by bin width and counts for total area = 1
        db = jnp.array(jnp.diff(bins), float)  # bin spacing
        counts = counts / db / counts.sum(axis=0)

    if reflect_infinities:
        counts = (
            counts[1:-1]
            + jnp.array([counts[0]] + [0] * (len(counts) - 3))
            + jnp.array([0] * (len(counts) - 3) + [counts[-1]])
        )
        if calc_sigma:
            sigma = (
                sigma[1:-1]
                + jnp.array([sigma[0]] + [0] * (len(sigma) - 3))
                + jnp.array([0] * (len(sigma) - 3) + [sigma[-1]])
            )

    if calc_sigma:
        return counts, sigma
    
    else:
        return counts, None