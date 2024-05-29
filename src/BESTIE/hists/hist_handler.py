

def hist_handler(config):
    binning_method = config["method"]

    if binning_method.lower() in ["binned_kde","bkde","binnedkde"]:
        calc_binning = None

    elif binning_method.lower() in ["softmax"]:
        raise NotImplementedError("Softmax binning is not yet implemented")
    
    elif binning_method.lower() in ["normalizing_flow"]:
        raise NotImplementedError("Normalizing flows are not yet implemented")
    
    return calc_binning