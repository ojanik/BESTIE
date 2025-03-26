import jax.numpy as jnp

def bin_loss(llh,injected_params,lss,aux,data_hist,sample_weights,**kwargs):
    mu, sigma = llh(injected_params,lss,aux,data_hist,sample_weights,skip_llh=True,**kwargs)
    ds_length = kwargs.pop("df_length")

    # Calculate sqrt(ssq)/mu ratio for each bin
    x = jnp.sqrt(sigma+1e-8)/(mu+1e-8) 
    # Calculate average loss over all populated bins
    loss = jnp.sum(x**2) / (jnp.sqrt(ds_length/jnp.sum(mu>0)))
    return loss
