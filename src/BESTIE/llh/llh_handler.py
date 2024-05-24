
def llh_handler(config):
    method = config["llh_method"]
    llh = None
    if method.lower() == "poisson":
        raise NotImplementedError("Poisson likelihood is not yet implemented")
    
    elif method.lower() == "say" or method.lower() == "effective":
        raise NotImplementedError("SAY likelihood is not yet implemented")
    
    else:
        raise NotImplementedError("The give likelihood method is not implemented")
    
    return llh