
def llh_handler(config):
    method = config["method"]
    llh = None
    if method.lower() == "poisson":
        from . import poisson_llh
        llh = poisson_llh
    
    elif method.lower() == "say" or method.lower() == "effective":
        raise NotImplementedError("SAY likelihood is not yet implemented")
    
    else:
        raise NotImplementedError("The give likelihood method is not implemented")
    
    return llh