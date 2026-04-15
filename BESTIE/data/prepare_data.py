import numpy as onp


import jax.numpy as jnp
Array = jnp.array

def create_input_data(df,config):
    mask_exists = onp.ones(len(df),dtype=bool)
    for exist in config["exists_vars"]:
        mask_exists &= onp.array(df[exist] == 1,dtype=bool)

    mask_cut = onp.ones(len(df),dtype=bool)
    
    for cut in config["cut_vars"]:
        print(f"Cutting {cut['var_name']} between {cut['min']} and {cut['max']}")
        mask_step = (Array(df[cut['var_name']] >= cut["min"])) 
        mask_step &= (Array(df[cut['var_name']] <= cut["max"]))
        print(f"From {cut['var_name']} cut {(mask_step).sum()} events passed")
        mask_cut &= mask_step

    mask = mask_exists & mask_cut

    output = []
    for vari in config["input_vars"]:
        dtemp = onp.array(df[vari["var_name"]])
        print(f"Adding {vari['var_name']} to input")
        if "scale" in vari:
            try:
                print(f"Scaling {vari['var_name']} with {vari['scale']}")
                dtemp = getattr(onp, vari["scale"])(dtemp)
            except:
                print("Couldn't find given scale method. Continuing without scaling the data.")

        if vari["transform"] in ["standardize"]:
            dtemp = (dtemp-onp.mean(dtemp[mask]))/onp.std(dtemp[mask])

        elif vari["transform"] in ["sphere"]:
            dtemp -= jnp.min(dtemp[mask]) 
            dtemp /= (jnp.max(dtemp[mask]) + 1e-3) #small constant to not get 1 as input value

        output.append(dtemp)
    
    output = onp.stack(output,axis=1)


    return output, mask

