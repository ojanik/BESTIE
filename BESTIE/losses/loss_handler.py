
import jax.numpy as jnp
Array = jnp.array

def loss_handler(config):
    lconfig = config["loss"]
    loss_method = lconfig["method"]
    losses = []
    loss_kwargs = {}
    if any(elem.lower() in ["fisher"] for elem in loss_method):

        optimality = lconfig["optimality"]

        if optimality.lower() in ["a","a_optimality","aoptimality"]:
            from .fisher_losses import A_optimality
            opti = A_optimality
        elif optimality.lower() in ["s","s_optimality","soptimality"]:
            from .fisher_losses import S_optimality
            opti = S_optimality
        elif optimality.lower() in ["d","d_optimality","doptimality","ellipsoid","uncertainty_ellipsoid","ellipsoid_volume","uncertainty_ellipsoid_volume"]:
            from .fisher_losses import D_optimality
            opti = D_optimality
        else:
            raise NotImplementedError(f"The {optimality} method for optimality is not yet implemented")
        
        loss_kwargs["opti"] = opti
        loss_kwargs["parameters_to_optimize"] = lconfig["parameters_to_optimize"]
        loss_kwargs["weight_norm"] = lconfig.get("weight_norm",None)
        loss_kwargs["rel_uncertainty_threshold"] = lconfig["soft_masking"].get("rel_uncertainty_threshold",None)
        loss_kwargs["mask_sharpness"] = lconfig["soft_masking"].get("mask_sharpness",None)

        from .fisher_losses import fisher_loss
        losses.append(fisher_loss)


    if any(elem.lower() in ["bin_loss","say_loss"] for elem in loss_method):
        from .bin_loss import bin_loss
        losses.append(bin_loss)
        loss_kwargs["df_length"] = config["dataset"]["length"]
        loss_kwargs["threshold"] = config["loss"]["MC_statistics_threshold"]

    if len(losses)==0:
        raise NotImplementedError("No valid loss method selected")
    
    else:
        def loss(mu,ssq,grad_hist,**kwargs):
            return Array([l(mu,ssq,grad_hist,**kwargs,**loss_kwargs) for l in losses])

    return loss


    
