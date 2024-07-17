


def loss_handler(config):

    loss_method = config["method"]

    if loss_method.lower() in ["fisher"]:

        optimality = config["optimality"]

        if optimality.lower() in ["a","a_optimality","aoptimality"]:
            from .fisher_losses import A_optimality
            opti = A_optimality
        elif optimality.lower() in ["s","s_optimality","soptimality"]:
            from .fisher_losses import S_optimality
            opti = S_optimality
        elif optimality.lower() in ["matthias","matthias_loss"]:
            from .fisher_losses import Matthias_loss
            opti = Matthias_loss
        else:
            raise NotImplementedError(f"The {optimality} method for optimality is not yet implemented")
        
        from jax import hessian
        def loss(llh,injected_params,lss,aux,data_hist,sample_weights):
            fish = hessian(llh)(injected_params,lss,aux,data_hist,sample_weights)
            return opti(fish,signal_idx=config["signal_idx"])

    else:
        raise NotImplementedError("The selected loss method is not implemented")
    
    return loss