from jax import random
import numpy as onp

def input_mapping(x, B):
  if B is None:
    return x
  else:
    x_proj = (2.*onp.pi*x) @ B.T
    return onp.concatenate([onp.sin(x_proj), onp.cos(x_proj)], axis=-1)

def get_B(config):
  
  B = None

  if "fourier_feature_mapping" in config:

    input_size = len(config["input_vars"])
    mapping_size = config["fourier_feature_mapping"]["mapping_size"]

    method = config["fourier_feature_mapping"]["method"]

    if method.lower() in ["gauss"]:
      scale = config["fourier_feature_mapping"]["scale"]

      rand_key = random.key(187)
      B = random.normal(rand_key, (mapping_size, input_size))
    
    else:
      B = None

  return B
    
