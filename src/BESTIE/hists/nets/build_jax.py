from flax import linen as nn
import jax.numpy as jnp
from .utilities import ResNetBlock_Dense
from . import utilities as uti

def build_jax(config):

    class model(nn.Module):
        
        @nn.compact
        def __call__(self, x):
            hidden_layers = config["hidden_layers"]
            for hidden_layer in hidden_layers:
                if hidden_layer["layer"].lower()=="resnet":
                    x = ResNetBlock_Dense(x,c_out=hidden_layer["size"],act_fn=getattr(uti,hidden_layer["activation"]))
                else:
                    x = getattr(nn,hidden_layer["layer"])(features=hidden_layer["size"],use_bias = "use_bias" in hidden_layer)(x)

                x = getattr(uti,hidden_layer["activation"])(x)


            return x
    return model
