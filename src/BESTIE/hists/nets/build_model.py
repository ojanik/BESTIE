from . import build_jax

def build_model(config):
    framework = config["framework"]

    if framework.lower() == "pytorch":
        raise NotImplementedError
    if framework.lower() == "jax":
        return build_jax.build_jax(config)
