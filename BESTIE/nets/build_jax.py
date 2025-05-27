from flax import linen as nn
import jax.numpy as jnp
from .utilities import ResNetBlock_Dense
from . import utilities as uti

def build_jax_dense(config):

    class model(nn.Module):
        
        @nn.compact
        def __call__(self, x,training=False):
            hidden_layers = config["hidden_layers"]
            for hidden_layer in hidden_layers:
                if hidden_layer["layer"].lower() == "resnet":
                    x = ResNetBlock_Dense(x, c_out=hidden_layer["size"],
                                           act_fn=getattr(uti, hidden_layer["activation"]))
                else:
                    x = getattr(nn, hidden_layer["layer"])(
                        features=hidden_layer["size"],
                        use_bias="use_bias" in hidden_layer
                    )(x)
                
                # Apply LayerNorm if requested

                
                # Apply activation
                x = getattr(uti, hidden_layer["activation"])(x)
                dropout_rate = hidden_layer.get("dropout",0.)
                x = nn.Dropout(rate=dropout_rate, deterministic=not training)(x)

                norm_type = hidden_layer.get("norm", None)
                if norm_type == "layernorm":
                    nn.LayerNorm(
                        epsilon=1e-6,
                        reduction_axes=-1,
                        feature_axes=-1
                    )(x)

            return x

    return model

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