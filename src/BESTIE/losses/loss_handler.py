


def loss_handler(config):

    loss_method = config["loss"]
    if loss_method.lower() in ["fisher"]:

        optimality = config["optimality"]

        if optimality.lower() in ["a","a_optimality","aoptimality"]
            from .fisher_losses import a_optimality
            opti = a_optimality

        else:
            raise NotImplementedError(f"The {optimality} method for optimality is not yet implemented")
        
        from jax import hessian
        def loss(llh,injected_params,lss,aux,data_hist):
            fish = hessian(llh)(injected_params,lss,aux,data_hist)
            return opti(fish,signal_idx=[0,1]) #TODO implement signal_idx readout from config

    else:
        raise NotImplementedError("The selected loss method is not implemented")
    
    return loss