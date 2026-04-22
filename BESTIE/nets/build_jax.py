from flax import linen as nn
import jax.numpy as jnp

from . import utilities as uti
from .utilities import ResNetDenseBlock, GatedDenseBlock  
from .fourier_feature_mapping import FourierFeatures


def build_jax_dense(config):

    class SubNet(nn.Module):
        hidden_layers: list

        @nn.compact
        def __call__(self, x, training: bool = False):
            for i, hl in enumerate(self.hidden_layers):
                layer_type = hl["layer"].lower()
                act_name = hl.get("activation", None)
                act_fn = getattr(uti, act_name) if act_name else None

                dropout_rate = hl.get("dropout", 0.0)
                norm_type = hl.get("norm", None)

                # ---- Block layers ----
                if layer_type == "resnet":
                    x = ResNetDenseBlock(
                        c_out=hl["size"],
                        act_fn=act_fn if act_fn is not None else nn.silu,
                        dropout=dropout_rate,
                        # optional: hl.get("use_post_add_activation", False)
                    )(x, training=training)

                    # IMPORTANT: no extra activation/dropout/norm here unless you want it
                    if hl.get("post_activation", False) and act_fn is not None:
                        x = act_fn(x)

                    if hl.get("post_norm", None) == "layernorm":
                        x = nn.LayerNorm(epsilon=1e-6)(x)

                    continue

                elif layer_type == "gated":
                    x = GatedDenseBlock(
                        c_out=hl["size"],
                        expansion=hl.get("expansion", 4),
                        dropout=dropout_rate,
                        # optional: act_gate=nn.silu
                    )(x, training=training)

                    if hl.get("post_activation", False) and act_fn is not None:
                        x = act_fn(x)

                    if hl.get("post_norm", None) == "layernorm":
                        x = nn.LayerNorm(epsilon=1e-6)(x)

                    continue

                # ---- Plain layers (Dense etc.) ----
                # Default to Dense if someone writes "dense"
                if layer_type in ("dense", "linear"):
                    x = nn.Dense(
                        features=hl["size"],
                        use_bias=hl.get("use_bias", True),
                        name=hl.get("name", f"dense_{i}")
                    )(x)
                else:
                    # Generic: assumes the linen module takes features=...
                    # (works for nn.Dense; be careful for layers that use different args)
                    x = getattr(nn, hl["layer"])(
                        features=hl["size"],
                        use_bias=hl.get("use_bias", True),
                    )(x)

                # Activation (for plain layers)
                if act_fn is not None:
                    x = act_fn(x)

                # Dropout (for plain layers)
                if dropout_rate > 0.0:
                    x = nn.Dropout(rate=dropout_rate, deterministic=not training)(x)

                # Optional LayerNorm (for plain layers)
                if norm_type == "layernorm":
                    x = nn.LayerNorm(epsilon=1e-6)(x)

            return x

    class Model(nn.Module):
        @nn.compact
        def __call__(self, x, training: bool = False):

            ff_cfg = config.get("fourier", None)

            if ff_cfg:
                sigmas = ff_cfg.get("sigmas")
                if sigmas is None:
                    sigmas = ff_cfg.get("sigma", 1.0)

                enc = FourierFeatures(
                    n_frequencies=ff_cfg.get("n_frequencies", 16),
                    sigmas=sigmas,
                    trainable=ff_cfg.get("trainable", False),
                )(x)  # list of (..., 2F)

                concat_raw = ff_cfg.get("concat_raw_input", False)

                shared_net = SubNet(
                    hidden_layers=config["hidden_layers"],
                    name="shared_subnet"
                )

                inp = jnp.concatenate([x, enc], axis=-1) if concat_raw else enc
                h = shared_net(inp, training=training)

            else:
                h = SubNet(hidden_layers=config["hidden_layers"])(x, training=training)

            # --- Main LSS head -----------------------------------------------
            # Optional projection head. When absent, the raw backbone output is
            # used as the LSS, exactly like before.
            proj_size = config.get("projection") or config.get("projection_size")
            if proj_size is not None:
                lss = nn.Dense(proj_size, name="projection")(h)
                proj_act = config.get("projection_activation", None)
                if proj_act:
                    lss = getattr(uti, proj_act)(lss)
            else:
                lss = h

            # --- Optional auxiliary score head -------------------------------
            # Opt-in via config["score_head"]. Shares the backbone `h` with the
            # LSS head and regresses a per-event score vector of dimension
            # n_params (one entry per parameter the aux loss supervises).
            # Returning a dict is the signal that the score head is active;
            # when absent we return just the LSS tensor so the rest of the
            # pipeline, evaluation, and checkpoints remain byte-equivalent
            # to pre-score-head behavior.
            sh_cfg = config.get("score_head", None)
            if sh_cfg is not None and sh_cfg.get("enabled", False):
                n_params = sh_cfg["n_params"]
                hidden = sh_cfg.get("hidden", []) or []
                act_name = sh_cfg.get("activation", "silu")
                act_fn = getattr(uti, act_name, None) or getattr(nn, act_name, nn.silu)

                s = h
                for i, hs in enumerate(hidden):
                    s = nn.Dense(hs, name=f"score_hidden_{i}")(s)
                    s = act_fn(s)
                score = nn.Dense(n_params, name="score_out")(s)
                return {"lss": lss, "score": score}

            return lss

    return Model


def build_MultiScaleFourierNetwork(config):
    DenseModel = build_jax_dense(config)

    class Model(nn.Module):
        @nn.compact
        def __call__(self, x, training: bool = False):
            # x is expected to be an iterable/list/array of frequency inputs
            dense_model = DenseModel(name="dense_model")

            ys = []
            for freq in x:
                ys.append(dense_model(freq, training=training))

            # Concatenate feature-wise unless you *really* want to stack batches
            h = jnp.concatenate(ys, axis=-1)

            # Optional final layer (kept from your original)
            last = config["hidden_layers"][-1]
            h = nn.Dense(
                features=last["size"],
                use_bias=last.get("use_bias", True),
                name="final_dense"
            )(h)
            h = getattr(uti, last["activation"])(h)
            return h

    return Model