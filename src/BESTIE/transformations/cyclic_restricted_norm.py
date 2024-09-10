import jax.numpy as jnp
import jax

def cyclic_restricted_norm(lss,**kwargs):
    try:
        lss = lss[:,0]
    except:
        pass
    lss0 = kwargs.get("lss0")
    #jax.debug.print("lss0 = {x}",x=lss0)
    psi = calc_psi(lss0)

    phi0 = kwargs.get("phi0")
    #jax.debug.print("phi0 = {x}",x=phi0)
    lss = 1/2 * (jnp.sin(lss/phi0+psi-1)+1)

    return lss

def calc_psi(lss0):
    return jnp.arcsin(2*lss0-1)

