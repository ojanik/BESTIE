from functools import partial
import jax.numpy as jnp

def hist_handler(config):
    method = config["method"]

    if method.lower() in ["binned_kde","bkde","binnedkde"]:
        from .bKDE import bKDE
        bandwidth = float(config["bandwidth"])
        bins = jnp.linspace(int(config["bins_low"]),int(config["bins_up"]),int(config["bins_number"])+1)
        calc_hist = partial(bKDE, bins=bins, bandwidth=bandwidth)
    
    elif method.lower() in ["tanh","tanhist"]:
        from .tanh_binning import tanhHist
        bins = jnp.linspace(int(config["bins_low"]),int(config["bins_up"]),int(config["bins_number"])+1)
        bandwidth = float(config["bandwidth"])
        calc_hist = partial(tanhHist, bins=bins, slope=bandwidth)

    elif method.lower() in ["tanh_nd","tanhist_nd","tanhnd"]:
        from .tanh_binning import tanhHistND
        bins_list = [jnp.linspace(int(config["bins_low"]),int(config["bins_up"]),int(config["bins_number"])+1) for _ in range(int(config["dim"]))]
        bandwidth = [float(config["bandwidth"]) for _ in range(int(config["dim"]))]
        calc_hist = partial(tanhHistND, bins_list=bins_list, slopes=bandwidth)

    elif method.lower() in ["softmax","vector","softmax_hist","vector_hist"]:
        from .vector_hist import vector_hist
        calc_hist = vector_hist
    
    elif method.lower() in ["normalizing_flow"]:
        raise NotImplementedError("Normalizing flows are not yet implemented")
    
    else:
        raise NotImplementedError(f"{method} is not implemented")
    
    return calc_hist