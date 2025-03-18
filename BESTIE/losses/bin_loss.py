import jax.numpy as jnp

def bin_loss(llh,injected_params,lss,aux,data_hist,sample_weights,**kwargs):
    mu, sigma = llh(injected_params,lss,aux,data_hist,sample_weights,skip_llh=True,**kwargs)
    df_length = kwargs.pop("df_length")

    x = jnp.sqrt(sigma+1e-8)/(mu+1) / (jnp.sqrt(df_length/jnp.sum(mu>0)))

    return jnp.sum(jnp.exp(abs(x))-1) 
