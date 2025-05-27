from . import build_jax
def model_handler(config):
    architecture = config["network"]["architecture"].lower()

    if architecture == "dense":
        from . import build_jax_dense
        return build_jax_dense(config["network"])
    elif architecture == "transformer":
        
        return build_jax.build_jax_transformer(config["network"])
    else:
        raise ValueError(f"Unknown architecture: {architecture}")