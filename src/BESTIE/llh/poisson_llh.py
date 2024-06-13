import jax.numpy as jnp
from jax.scipy.special import gammaln


def poisson_llh(k, mu):
    mu = mu + 1e-12
    llh = jnp.where(mu>0,
                    k * jnp.log(mu) - mu - gammaln(k + 1),
                    jnp.where(k>0,
                              -690*k,
                              0))
    return llh