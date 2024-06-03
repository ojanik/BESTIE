from functools import partial

def hist_handler(config):
    method = config["method"]

    if method.lower() in ["binned_kde","bkde","binnedkde"]:
        from .bKDE import bKDE
        calc_hist = partial(bKDE, bandwidth=config["bandwidth"])
        
    elif method.lower() in ["softmax"]:
        raise NotImplementedError("Softmax binning is not yet implemented")
    
    elif method.lower() in ["normalizing_flow"]:
        raise NotImplementedError("Normalizing flows are not yet implemented")
    
    return calc_hist