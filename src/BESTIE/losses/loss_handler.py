


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
        
        
    
        if config["fisher_method"].lower() in ["jacobian","expectation"]:
            from jax import grad, jacfwd, vmap
            import jax
            import jax.numpy as jnp 
            def loss(llh,injected_params,lss,aux,data_hist,sample_weights,**kwargs):
                grads = jacfwd(llh)(injected_params,lss,aux,data_hist,sample_weights,skip_llh=True,**kwargs)
                fish_i = vmap(lambda g: jnp.outer(g, g))(grads)
                hist_counts = llh(injected_params,lss,aux,data_hist,sample_weights,skip_llh=True,**kwargs) + 1e-8
                nonzero_hist_counts = hist_counts != 0
                fish_i = jnp.where(nonzero_hist_counts[:, None, None],
                                    fish_i / hist_counts[:, None, None], 
                                    jnp.zeros(fish_i.shape))  # Shape: (1650, 11, 11)
                fish = jnp.sum(fish_i, axis=0) * -1.
                return opti(fish,signal_idx=config["signal_idx"])
        
        else:
            from jax import hessian, grad, jacfwd, vmap
            import jax.numpy as jnp
            import jax
            import numpy as onp
            def loss(llh,injected_params,lss,aux,data_hist,sample_weights,**kwargs):
                #grads = grad(llh)(injected_params,lss,aux,data_hist,sample_weights,**kwargs)
                #fish = -1 * jnp.outer(grads,grads)
                #jax.debug.print("fish grads: {fish}",fish=fish)
                fish = hessian(llh)(injected_params,lss,aux,data_hist,sample_weights,**kwargs)
                """jax.debug.print("fish hess: {fish}",fish=fish)
                jax.debug.print("fish hess shape: {fish}",fish=fish.shape)
                jax.debug.print("fish hess eigenvalues: {x}", x= jnp.linalg.eigvals(fish))


                grads = jacfwd(llh)(injected_params,lss,aux,data_hist,sample_weights,skip_llh=True,**kwargs)
                fish_i = vmap(lambda g: jnp.outer(g, g))(grads)
                hist_counts = llh(injected_params,lss,aux,data_hist,sample_weights,skip_llh=True,**kwargs)
                nonzero_hist_counts = hist_counts != 0
                fish_i = jnp.where(nonzero_hist_counts[:, None, None],
                                    fish_i / hist_counts[:, None, None], 
                                    jnp.zeros(fish_i.shape))  # Shape: (1650, 11, 11)
                fish = jnp.sum(fish_i, axis=0) * -1.

                jax.debug.print("fish grad: {fish}",fish=fish)
                jax.debug.print("fish grad shape: {fish}",fish=fish.shape)
                jax.debug.print("fish grad eigenvalues: {x}", x= jnp.linalg.eigvals(-1. * fish))"""
                return opti(fish,signal_idx=config["signal_idx"])

    elif loss_method.lower() in ["scan"]:
        raise NotImplementedError("Scan loss is currently not fully implemented")
        from .scan_loss import calc_scan_loss

        def loss(llh,injected_params,lss,aux,data_hist,sample_weights,**kwargs):
            return calc_scan_loss(llh,injected_params,lss,aux,data_hist,sample_weights,scan_parameter_idx=config["signal_idx"],**kwargs)

    else:
        raise NotImplementedError("The selected loss method is not implemented")
    
    return loss