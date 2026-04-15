

def transformation_handler(config):
    method = config["method"]

    if method.lower() in ["no_scaling","no","none"]:
        from .no_scaling import no_scaling
        print("Using no_scaling as transformation")
        transformation = no_scaling

    elif method.lower() in ["linear","linear_scaling","01_scaling"]:
        from .linear_scaling import linear_scaling
        transformation = linear_scaling

    elif method.lower() in ["cyclic","cylclic_restricted","cyclic_restricted_norm"]:
        from .cyclic_restricted_norm import cyclic_restricted_norm
        transformation = cyclic_restricted_norm


    else:
        raise NotImplementedError("This method of transformation is not yet implemented")

    return transformation