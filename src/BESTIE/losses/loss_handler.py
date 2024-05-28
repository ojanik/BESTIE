
def loss_handler(config):

    loss_method = config["loss"]

    if loss_method.lower() == "a_optimality":
        from .fisher_losses import a_optimality
        loss = a_optimality
    
    else:
        raise NotImplementedError("The selected loss method is not implemented")
    
    return loss