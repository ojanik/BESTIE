import jax.numpy as jnp

def standard_binning(energy,coszenith,weights):
    bins_energy = jnp.logspace(2,7,51)
    bins_coszenith = jnp.linspace(-1,0.08,34)

    H,_,_ = jnp.histogram2d(energy,coszenith,bins=[bins_energy,bins_coszenith],weights=weights)

    return H.flatten()