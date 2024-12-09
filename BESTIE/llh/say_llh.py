import jax.numpy as jnp
from jax.scipy.special import gammaln


def say_llh(k, mu, sigma):
    mu = mu + 1e-8
    k = k + 1e-8
    sigma = sigma + 1e-8

    a = 1.
    b = 0.

    alpha = mu**2 / sigma ** 2 + a
    beta = mu / sigma**2 + b

    llh = jnp.where(mu>0,
                    alpha * jnp.log(beta) - (k+alpha)*jnp.log(1+beta) + gammaln(k+alpha)- gammaln(alpha) - gammaln(k+1) ,
                    jnp.where(k>0,
                              -690*k,
                              0))
    return llh