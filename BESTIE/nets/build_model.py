def model_handler(config):
    architecture = config["architecture"].lower()

    if architecture == "dense":
        from . import build_jax_dense
        return build_jax_dense(config)
    elif architecture == "transformer":
        from . import build_jax_transformer
        return build_jax_transformer(config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")