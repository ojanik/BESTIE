from jax import vmap

def weight_handler(config):
    method = config["weighting"]["method"]

    if method.lower() in ["nnm_fit","nnmfit","nnm"]:
        # the NNMFit config should be setup in such a way that only the weight of ONE event is calculated
        # the function is then vectorized using vmap
        # this makes handling the batch_sizes easier
        from ..NNMFit_handlers import NNMFit_handler
        handler = NNMFit_handler(config)
        w_fn = handler.get_weight_function()
        w_fn = vmap(w_fn)

    else:
        raise NotImplementedError(f"The weighting method {str(method)} has not be implemented")
    
    return w_fn