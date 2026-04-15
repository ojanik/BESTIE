from src import BESTIE
import jax.numpy as jnp
from jax import random

config = {"method": "bkdE","bandwidth":0.01}
hist = BESTIE.hists.hist_handler(config)

bins = jnp.linspace(0,1,100)
key = random.key(33)
k = random.uniform(key,(10_000,))

kBKDE = hist(k,bins=bins,weights=jnp.ones(len(k)))

assert len(kBKDE) == len(bins)