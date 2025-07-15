from . import build_jax
def model_handler(config):
    model_dict = {}

    for hkey in config["hists"].keys():
        model_dict[hkey] = {}
        architecture = config["hists"][hkey]["network"]["architecture"].lower()

        if architecture == "dense":
            from . import build_jax_dense
            model = build_jax_dense(config["hists"][hkey]["network"])
        elif architecture == "transformer":
            
            model = build_jax.build_jax_transformer(config["hists"][hkey]["network"])
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        model_dict[hkey]["model"] = model

    return model_dict