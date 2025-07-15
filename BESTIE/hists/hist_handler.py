from functools import partial
import jax.numpy as jnp

def hist_handler(config):
    histmaker_dict = {}
    for hkey in config["hists"].keys():
        print(f"Creating hist function for {hkey}")
        hconfig = config["hists"][hkey]["hists"]
        histmaker_dict[hkey] = {}

        method = hconfig["method"]

        if method.lower() in ["binned_kde","bkde","binnedkde"]:
            from .bKDE import bKDE
            bandwidth = float(hconfig["bandwidth"])
            bins = jnp.linspace(int(hconfig["bins_low"]),int(hconfig["bins_up"]),int(hconfig["bins_number"])+1)
            calc_hist = partial(bKDE, bins=bins, bandwidth=bandwidth)
        
        elif method.lower() in ["tanh","tanhist"]:
            from .tanh_binning import tanhHist
            bins = jnp.linspace(int(hconfig["bins_low"]),int(hconfig["bins_up"]),int(hconfig["bins_number"])+1)
            bandwidth = float(hconfig["bandwidth"])
            calc_hist = partial(tanhHist, bins=bins, slope=bandwidth)

        elif method.lower() in ["tanh_nd","tanhist_nd","tanhnd"]:
            from .tanh_binning import tanhHistND
            bins_list = [jnp.linspace(int(hconfig["bins_low"]),int(hconfig["bins_up"]),int(hconfig["bins_number"])+1) for _ in range(int(hconfig["dim"]))]
            bandwidth = [float(hconfig["bandwidth"]) for _ in range(int(hconfig["dim"]))]
            calc_hist = partial(tanhHistND, bins_list=bins_list, slopes=bandwidth)
        
        elif method.lower() in ["normalizing_flow"]:
            raise NotImplementedError("Normalizing flows are not yet implemented")
        
        else:
            raise NotImplementedError(f"{method} is not implemented")
        
        histmaker_dict[hkey]["calc_hist"] = calc_hist
    
    return histmaker_dict