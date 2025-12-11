from flax import linen as nn
import jax.numpy as jnp

def ResNetBlock_Dense(x,c_out,act_fn=nn.relu):
    skip = x
    z = nn.Dense(c_out)(x)
    #z = nn.BatchNorm()(z, use_running_average=False)
    z = act_fn(z)
    z = nn.Dense(c_out)(z)
    #z = nn.BatchNorm()(z, use_running_average=False)


    x_out = act_fn(z + skip)
    return x_out

def Gated_Dense(x,c_out,act_fn=nn.relu):
    skip = x
    z = nn.Dense(c_out)(x)
    z = nn.LayerNorm(
                        epsilon=1e-6,
                        reduction_axes=-1,
                        feature_axes=-1
                    )(z)
    z = act_fn(z)
    u,v = jnp.split(z,2,axis=-1)
    
    v = nn.LayerNorm(
                        epsilon=1e-6,
                        reduction_axes=-1,
                        feature_axes=-1
                    )(v)
    v = nn.Dense(int(c_out/2),kernel_init=nn.initializers.zeros, bias_init=nn.initializers.ones)(v)

    z = jnp.multiply(u,v)

    z = nn.Dense(c_out, kernel_init=nn.initializers.zeros)(z)
    #z = nn.BatchNorm()(z, use_running_average=False)


    x_out = act_fn(z + skip)
    return x_out

def sin(x):
    return jnp.sin(x)

def cos(x):
    return jnp.cos(x)

class sincos(nn.Module):
    

    def setup(self):
        self.alpha = self.param('alpha', nn.initializers.normal(stddev=1),
                                 (1,))

    def __call__(self, X):
        return jnp.sqrt(self.alpha) * jnp.sin(X) + jnp.sqrt(1-self.alpha) * jnp.cos(X)

def sawtooth(x):
    return x%1

def relu(x):
    return nn.relu(x)

def elu(x):
    return nn.elu(x)

def gelu(x):
    return nn.gelu(x)

def silu(x):
    return nn.silu(x)

def softmax(x):
    return nn.softmax(x)

def sigmoid(x):
    return nn.sigmoid(x)

def lin(x):
    return x

def hard_tanh(x):
    return nn.hard_tanh(x)

def tanh(x):
    return jnp.tanh(x)

