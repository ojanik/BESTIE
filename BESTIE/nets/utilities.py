from flax import linen as nn
import jax.numpy as jnp

def ResNetBlock_Dense(x,c_out,act_fn=nn.relu):
    z = nn.Dense(c_out)(x)
    #z = nn.BatchNorm()(z, use_running_average=False)
    z = act_fn(z)
    z = nn.Dense(c_out)(z)
    #z = nn.BatchNorm()(z, use_running_average=False)


    x_out = act_fn(z + x)
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
