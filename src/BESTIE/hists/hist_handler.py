from functools import partial
import jax.numpy as jnp

def hist_handler(config):
    method = config["method"]

    if method.lower() in ["binned_kde","bkde","binnedkde"]:
        from .bKDE import bKDE
        bandwidth = float(config["bandwidth"])
        bins = jnp.linspace(int(config["bins_low"]),int(config["bins_up"]),int(config["bins_number"])+1)
        calc_hist = partial(bKDE, bins=bins, bandwidth=bandwidth)
        
    elif method.lower() in ["softmax"]:
        raise NotImplementedError("Softmax binning is not yet implemented")
    
    elif method.lower() in ["normalizing_flow"]:
        raise NotImplementedError("Normalizing flows are not yet implemented")
    
    return calc_hist