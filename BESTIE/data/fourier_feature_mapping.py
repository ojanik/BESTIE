from jax import random
import numpy as onp
import jax.numpy as jnp

def input_mapping(x, B, logscale=0.):
  if B is None:
    return x
  else:
    B = B * (10**logscale)
    x_proj = (2.*jnp.pi*x) @ B.T
    return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

def get_B(config):
  
  B = None

  if "fourier_feature_mapping" in config:

    input_size = len(config["input_vars"])
    mapping_size = config["fourier_feature_mapping"]["mapping_size" ]

    method = config["fourier_feature_mapping"]["method"]

    if method.lower() in ["gauss"]:
      rand_key = random.key(187)
      B = random.normal(rand_key, (mapping_size, input_size))
    
    else:
      B = None

  return B
    
