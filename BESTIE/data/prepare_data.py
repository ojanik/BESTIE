import numpy as onp


import jax.numpy as jnp
Array = jnp.array

def create_input_data(df,config):
    mask_exists = onp.ones(len(df),dtype=bool)
    for exist in config["exists_vars"]:
        mask_exists &= onp.array(df[exist] == 1,dtype=bool)

    mask_cut = onp.ones(len(df),dtype=bool)
    
    for cut in config["cut_vars"]:
        mask_cut &= (Array(df[cut["var_name"]] >= cut["min"])) 
        mask_cut &= (Array(df[cut["var_name"]] <= cut["max"]))

    mask = mask_exists & mask_cut

    output = []
    for vari in config["input_vars"]:
        dtemp = onp.array(df[vari["var_name"]])

        if "scale" in vari:
            try:
                dtemp = getattr(onp, vari["scale"])(dtemp)
            except:
                print("Couldn't find given scale method. Continuing without scaling the data.")

        if "standardize" in vari:
            print("The standardize key in the dataset config is deprecated, please use transform")
            dtemp = (dtemp-onp.mean(dtemp[mask]))/onp.std(dtemp[mask])

        if vari["transform"] in ["standardize"]:
            dtemp = (dtemp-onp.mean(dtemp[mask]))/onp.std(dtemp[mask])

        elif vari["transform"] in ["sphere"]:
            dtemp -= jnp.min(dtemp[mask])
            dtemp /= (jnp.max(dtemp[mask]) + 1e-3) #small constant to not get 1 as input value

        output.append(dtemp)
    
    output = onp.stack(output,axis=1)


    return output, mask_exists, mask_cut

def calc_bin_idx(data):
    energy = Array(data[:,0])
    coszenith = jnp.cos(Array(data[:,1]))

    energy_bins = jnp.logspace(2,7,51)
    coszenith_bins = jnp.linspace(-1,0.0872,34)

    energy_digi = jnp.digitize(energy,energy_bins) - 1
    zenith_digi = jnp.digitize(coszenith,coszenith_bins) - 1
    bins_flattened = energy_digi * 33 + zenith_digi

    return bins_flattened