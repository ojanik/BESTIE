from src.BESTIE.llh import llh_handler
import jax.numpy as jnp
from jax import random

## Add here the llh method you want to test

config = {"llh_method": "poisson"}
llh = llh_handler(config)

# create and bin some random data
bins = jnp.linspace(0,1,100)
key = random.key(33)
k = random.uniform(key,(10_000,))
k, _ = jnp.histogram(k, bins)
key, subkey = random.split(key)
mu = random.uniform(key,(10_000,))
mu, _ = jnp.histogram(mu,bins)

print("llh calculated as: ",llh(k,mu).sum())