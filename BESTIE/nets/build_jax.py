from flax import linen as nn
import jax.numpy as jnp

from . import utilities as uti
from .utilities import ResNetDenseBlock, GatedDenseBlock  
from .fourier_feature_mapping import FourierFeatures #, MultiScaleFourierFeatures


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

                # outs = []
                # for enc in enc_list:
                #     inp = jnp.concatenate([x, enc], axis=-1) if concat_raw else enc
                #     outs.append(shared_net(inp, training=training))
                inp = jnp.concatenate([x, enc], axis=-1) if concat_raw else enc
                x = shared_net(inp, training=training)

                #x = jnp.concatenate(x, axis=-1)

            else:
                x = SubNet(hidden_layers=config["hidden_layers"])(x, training=training)

            # Optional projection head
            proj_size = config.get("projection") or config.get("projection_size")
            if proj_size is not None:
                x = nn.Dense(proj_size, name="projection")(x)
                proj_act = config.get("projection_activation", None)
                if proj_act:
                    x = getattr(uti, proj_act)(x)

            return x

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

# from flax import linen as nn
# import jax.numpy as jnp
# from .utilities import ResNetBlock_Dense, Gated_Dense
# from . import utilities as uti
# from .fourier_feature_mapping import FourierFeatures, MultiScaleFourierFeatures


# def build_jax_dense(config):

#     class SubNet(nn.Module):
#         hidden_layers: list

#         @nn.compact
#         def __call__(self, x, training: bool = False):
#             for hl in self.hidden_layers:
#                 if hl["layer"].lower() == "resnet":
#                     x = ResNetBlock_Dense(
#                         x,
#                         c_out=hl["size"],
#                         act_fn=getattr(uti, hl["activation"]),
#                     )
#                 elif hl["layer"].lower() == "gated":
#                     x = Gated_Dense(
#                         x,
#                         c_out=hl["size"],
#                         act_fn=getattr(uti, hl["activation"]),
#                     )
#                 else:
#                     x = getattr(nn, hl["layer"])(
#                         features=hl["size"],
#                         use_bias = hl.get("use_bias", True),
#                     )(x)

#                 # Activation
#                 x = getattr(uti, hl["activation"])(x)

#                 # Dropout
#                 x = nn.Dropout(
#                     rate=hl.get("dropout", 0.0),
#                     deterministic=not training
#                 )(x)

#                 # Optional LayerNorm (assign back!)
#                 if hl.get("norm", None) == "layernorm":
#                     x = nn.LayerNorm(
#                         epsilon=1e-6,
#                         reduction_axes=-1,
#                         feature_axes=-1
#                     )(x)
#             return x

#     class model(nn.Module):
#         @nn.compact
#         def __call__(self, x, training: bool = False):
#             # ----- Optional multi-scale Fourier front-end (single-scale = [sigma]) -----
#             ff_cfg = config.get("fourier", None)
#             if ff_cfg:
#                 sigmas = ff_cfg.get("sigmas")
#                 if sigmas is None:
#                     sigmas = [ff_cfg.get("sigma", 1.0)]

#                 enc_list = MultiScaleFourierFeatures(
#                     n_frequencies=ff_cfg.get("n_frequencies", 64),
#                     sigmas=sigmas,
#                     trainable=ff_cfg.get("trainable", False),
#                 )(x)  # list of (..., 2F)

#                 concat_raw = ff_cfg.get("concat_raw_input", False)
#                 shared_net = SubNet(hidden_layers=config["hidden_layers"], name="shared_subnet")

#                 outs = []
#                 for enc in enc_list:
#                     inp = jnp.concatenate([x, enc], axis=-1) if concat_raw else enc
#                     outs.append(shared_net(inp, training=training))

#                 x = jnp.concatenate(outs, axis=-1)

#             else:
#                 # ----- Plain base network path (no Fourier) -----
#                 x = SubNet(hidden_layers=config["hidden_layers"])(x, training=training)

#             # ----- Optional projection head (applied in BOTH cases) -----
#             proj_size = config.get("projection") or config.get("projection_size")
#             if proj_size is not None:
#                 x = nn.Dense(proj_size, name="projection")(x)

#                 proj_act = config.get("projection_activation", None)
#                 if proj_act:
#                     x = getattr(uti, proj_act)(x)


#             return x

#     return model


# def build_MultiScaleFourierNetwork(config):
#     dense_model = build_jax_dense(config)()
#     class model(nn.Module):
#         @nn.compact
#         def __call__(self,x):
#             y = []
#             for frequency in x:
#                 y.append(dense_model(frequency))
#             hidden_layer = config["hidden_layers"][-1]
#             x = jnp.concatenate(y)
#             x = nn.Dense(features=hidden_layer["size"],use_bias="use_bias" in hidden_layer)(x)
#             x = getattr(uti, hidden_layer["activation"])(x)
#             return x
#     return model


# def build_jax_transformer(config):
#     class TransformerEncoderBlock(nn.Module):
#         embed_dim: int
#         num_heads: int
#         mlp_dim: int

#         @nn.compact
#         def __call__(self, x):
#             # Self-attention
#             x_attn = nn.SelfAttention(num_heads=self.num_heads)(x)
#             x = nn.LayerNorm()(x + x_attn)

#             # Feed-forward MLP
#             y = nn.Dense(self.mlp_dim)(x)
#             y = nn.relu(y)
#             y = nn.Dense(self.embed_dim)(y)
#             x = nn.LayerNorm()(x + y)
#             return x

#     class Transformer(nn.Module):
#         embed_dim = config["embed_dim"]
#         num_heads = config["num_heads"]
#         mlp_dim = config["mlp_dim"]
#         num_layers = config["num_layers"]

#         @nn.compact
#         def __call__(self, x):
#             # x: (batch_size, k_events, n_features)
#             x = nn.Dense(self.embed_dim)(x)

#             for _ in range(self.num_layers):
#                 x = TransformerEncoderBlock(
#                     embed_dim=self.embed_dim,
#                     num_heads=self.num_heads,
#                     mlp_dim=self.mlp_dim
#                 )(x)

#             # Project each event embedding into scalar
#             summary = nn.Dense(1)(x).squeeze(-1)
#             return summary  # (batch_size, k_events)

#     return Transformer