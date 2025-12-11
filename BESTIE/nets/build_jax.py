from flax import linen as nn
import jax.numpy as jnp
from .utilities import ResNetBlock_Dense, Gated_Dense
from . import utilities as uti
from .fourier_feature_mapping import FourierFeatures, MultiScaleFourierFeatures


def build_jax_dense(config):

    class SubNet(nn.Module):
        hidden_layers: list

        @nn.compact
        def __call__(self, x, training: bool = False):
            for hl in self.hidden_layers:
                if hl["layer"].lower() == "resnet":
                    x = ResNetBlock_Dense(
                        x,
                        c_out=hl["size"],
                        act_fn=getattr(uti, hl["activation"]),
                    )
                elif hl["layer"].lower() == "gated":
                    x = Gated_Dense(
                        x,
                        c_out=hl["size"],
                        act_fn=getattr(uti, hl["activation"]),
                    )
                else:
                    x = getattr(nn, hl["layer"])(
                        features=hl["size"],
                        use_bias=("use_bias" in hl),
                    )(x)

                # Activation
                x = getattr(uti, hl["activation"])(x)

                # Dropout
                x = nn.Dropout(
                    rate=hl.get("dropout", 0.0),
                    deterministic=not training
                )(x)

                # Optional LayerNorm (assign back!)
                if hl.get("norm", None) == "layernorm":
                    x = nn.LayerNorm(
                        epsilon=1e-6,
                        reduction_axes=-1,
                        feature_axes=-1
                    )(x)
            return x

    class model(nn.Module):
        @nn.compact
        def __call__(self, x, training: bool = False):
            # ----- Optional multi-scale Fourier front-end (single-scale = [sigma]) -----
            ff_cfg = config.get("fourier", None)
            if ff_cfg:
                sigmas = ff_cfg.get("sigmas")
                if sigmas is None:
                    sigmas = [ff_cfg.get("sigma", 1.0)]

                enc_list = MultiScaleFourierFeatures(
                    n_frequencies=ff_cfg.get("n_frequencies", 64),
                    sigmas=sigmas,
                    trainable=ff_cfg.get("trainable", False),
                )(x)  # list of (..., 2F)

                concat_raw = ff_cfg.get("concat_raw_input", False)
                shared_net = SubNet(hidden_layers=config["hidden_layers"], name="shared_subnet")

                outs = []
                for enc in enc_list:
                    inp = jnp.concatenate([x, enc], axis=-1) if concat_raw else enc
                    outs.append(shared_net(inp, training=training))

                x = jnp.concatenate(outs, axis=-1)

            else:
                # ----- Plain base network path (no Fourier) -----
                x = SubNet(hidden_layers=config["hidden_layers"])(x, training=training)

            # ----- Optional projection head (applied in BOTH cases) -----
            proj_size = config.get("projection") or config.get("projection_size")
            if proj_size is not None:
                x = nn.Dense(proj_size, name="projection")(x)

                proj_act = config.get("projection_activation", None)
                if proj_act:
                    x = getattr(uti, proj_act)(x)


            return x

    return model

# def build_jax_dense(config):

#     class model(nn.Module):
        
#         @nn.compact
#         def __call__(self, x,training=False):

#             if "fourier" in config:
#                 ff_cfg = config["fourier"]
#                 x = FourierFeatures(
#                     n_frequencies=ff_cfg.get("n_frequencies", 64),
#                     sigma=ff_cfg.get("sigma", 1.0),
#                     trainable=ff_cfg.get("trainable", False),
#                 )(x)

#             hidden_layers = config["hidden_layers"]
#             for hidden_layer in hidden_layers:
#                 if hidden_layer["layer"].lower() == "resnet":
#                     x = ResNetBlock_Dense(x, c_out=hidden_layer["size"],
#                                            act_fn=getattr(uti, hidden_layer["activation"]))
#                 else:
#                     x = getattr(nn, hidden_layer["layer"])(
#                         features=hidden_layer["size"],
#                         use_bias="use_bias" in hidden_layer
#                     )(x)
                
#                 # Apply LayerNorm if requested

                
#                 # Apply activation
#                 x = getattr(uti, hidden_layer["activation"])(x)
#                 dropout_rate = hidden_layer.get("dropout",0.)
#                 x = nn.Dropout(rate=dropout_rate, deterministic=not training)(x)

#                 norm_type = hidden_layer.get("norm", None)
#                 if norm_type == "layernorm":
#                     nn.LayerNorm(
#                         epsilon=1e-6,
#                         reduction_axes=-1,
#                         feature_axes=-1
#                     )(x)

#             return x

#     return model

def build_MultiScaleFourierNetwork(config):
    dense_model = build_jax_dense(config)()
    class model(nn.Module):
        @nn.compact
        def __call__(self,x):
            y = []
            for frequency in x:
                y.append(dense_model(frequency))
            hidden_layer = config["hidden_layers"][-1]
            x = jnp.concatenate(y)
            x = nn.Dense(features=hidden_layer["size"],use_bias="use_bias" in hidden_layer)(x)
            x = getattr(uti, hidden_layer["activation"])(x)
            return x
    return model


def build_jax_transformer(config):
    class TransformerEncoderBlock(nn.Module):
        embed_dim: int
        num_heads: int
        mlp_dim: int

        @nn.compact
        def __call__(self, x):
            # Self-attention
            x_attn = nn.SelfAttention(num_heads=self.num_heads)(x)
            x = nn.LayerNorm()(x + x_attn)

            # Feed-forward MLP
            y = nn.Dense(self.mlp_dim)(x)
            y = nn.relu(y)
            y = nn.Dense(self.embed_dim)(y)
            x = nn.LayerNorm()(x + y)
            return x

    class Transformer(nn.Module):
        embed_dim = config["embed_dim"]
        num_heads = config["num_heads"]
        mlp_dim = config["mlp_dim"]
        num_layers = config["num_layers"]

        @nn.compact
        def __call__(self, x):
            # x: (batch_size, k_events, n_features)
            x = nn.Dense(self.embed_dim)(x)

            for _ in range(self.num_layers):
                x = TransformerEncoderBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_dim=self.mlp_dim
                )(x)

            # Project each event embedding into scalar
            summary = nn.Dense(1)(x).squeeze(-1)
            return summary  # (batch_size, k_events)

    return Transformer