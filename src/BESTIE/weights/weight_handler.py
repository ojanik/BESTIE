

def weight_handler(config):
    method = config["weighting"]["method"]

    if method.lower() in ["nnm_fit","nnmfit","nnm"]:
        from ..NNMFit_handlers import NNMFit_handler
        handler = NNMFit_handler(config)
        w_fn = handler.get_weight_function()

    else:
        raise NotImplementedError(f"The weighting method {str(method)} has not be implemented")
    
    return w_fn