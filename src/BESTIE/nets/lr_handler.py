

def lr_handler(config,steps_per_epoch):
    method = config["training"]["lr"]["method"]

    if method.lower() in ["constant","scalar"]:
        lr = config["training"]["lr"]["lr"]

    elif method.lower() in ["cosine","cos"]:
        from .learning_rates import create_cosine_lr
        lr = create_cosine_lr(config["training"]["lr"]["lr"],
                              steps_per_epoch=steps_per_epoch,
                              num_epochs=config["training"]["epochs"],
                              warmup_epochs=0)
    
    else:
        raise NotImplementedError(f"lr method {method} is not implemented")
    
    return lr