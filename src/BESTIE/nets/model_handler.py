from . import build_jax

def model_handler(config):
    config = config["network"]
    framework = config["framework"]

    if framework.lower() == "pytorch":
        raise NotImplementedError
    if framework.lower() == "jax":
        return build_jax.build_jax(config)