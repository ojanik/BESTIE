


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
        elif optimality.lower() in ["d","d_optimality","doptimality","ellipsoid","uncertainty_ellipsoid","ellipsoid_volume","uncertainty_ellipsoid_volume"]:
            from .fisher_losses import D_optimality
            opti = D_optimality
        else:
            raise NotImplementedError(f"The {optimality} method for optimality is not yet implemented")
        
        
    
        if config["fisher_method"].lower() in ["jacobian","expectation"]:
            from jax import grad, jacfwd, vmap
            import jax
            import jax.numpy as jnp 
            from BESTIE.utilities import rearrange_matrix
            def loss(llh,injected_params,lss,aux,data_hist,sample_weights,**kwargs):
                grads = jacfwd(llh)(injected_params,lss,aux,data_hist,sample_weights,skip_llh=True,**kwargs)
                fish_i = vmap(lambda g: jnp.outer(g, g))(grads)
                hist_counts = llh(injected_params,lss,aux,data_hist,sample_weights,skip_llh=True,**kwargs) + 1e-8
                nonzero_hist_counts = hist_counts != 0
                fish_i = jnp.where(nonzero_hist_counts[:, None, None],
                                    fish_i / hist_counts[:, None, None], 
                                    jnp.zeros(fish_i.shape))  # Shape: (1650, 11, 11)
                fish = jnp.sum(fish_i, axis=0) 

                # marginalize fisher information
                fish = rearrange_matrix(fish, config["signal_idx"])
                k = len(config["signal_idx"])
                A = fish[:k, :k]
                B = fish[:k, k:]
                C = fish[k:, k:]

                # Compute the inverse of C
                C_inv = jnp.linalg.inv(C)
                
                # Compute the Schur complement S = A - B * C_inv * B^T
                S = A - B @ C_inv @ B.T
                return opti(S,signal_idx=config["signal_idx"])
        
        else:
            from jax import hessian, grad, jacfwd, vmap
            import jax.numpy as jnp
            import jax
            import numpy as onp
            from BESTIE.utilities import rearrange_matrix
            def loss(llh,injected_params,lss,aux,data_hist,sample_weights,**kwargs):
                #grads = grad(llh)(injected_params,lss,aux,data_hist,sample_weights,**kwargs)
                #fish = -1 * jnp.outer(grads,grads)
                #jax.debug.print("fish grads: {fish}",fish=fish)
                fish = hessian(llh)(injected_params,lss,aux,data_hist,sample_weights,**kwargs)

                # marginalize fisher information
                fish = rearrange_matrix(fish, config["signal_idx"])
                k = len(config["signal_idx"])
                A = fish[:k, :k]
                B = fish[:k, k:]
                C = fish[k:, k:]

                # Compute the inverse of C
                C_inv = onp.linalg.inv(C)

                # Compute the Schur complement S = A - B * C_inv * B^T
                S = A - B @ C_inv @ B.T

                return opti(S)

    elif loss_method.lower() in ["scan"]:
        raise NotImplementedError("Scan loss is currently not fully implemented")
        from .scan_loss import calc_scan_loss

        def loss(llh,injected_params,lss,aux,data_hist,sample_weights,**kwargs):
            return calc_scan_loss(llh,injected_params,lss,aux,data_hist,sample_weights,scan_parameter_idx=config["signal_idx"],**kwargs)

    if any(elem.lower() in ["bin_loss","say_loss"] for elem in loss_method):
        from .bin_loss import bin_loss
        losses.append(bin_loss)
        loss_kwargs["df_length"] = config["dataset"]["length"]


    else:
        raise NotImplementedError("The selected loss method is not implemented")
    
    return loss


    
