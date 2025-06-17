import numpy as onp
import pandas as pd
import jax.numpy as jnp
Array = jnp.array

import os

from BESTIE.utilities import parse_yaml
from BESTIE.data import create_input_data, calc_bin_idx, compute_knn_weights

"""
def make_torch_dataset(config,df,weighter=None):

    # TODO implement use of weighter to caclulate expected weights

    


    input_data, mask_exists, mask_cut = create_input_data(df,config["dataset"])
    input_data = input_data[mask_exists&mask_cut]
    print("NaNs in input data: ",jnp.isnan(input_data).sum())
    print("input data only contains finite values: ",jnp.isfinite(input_data).all())

    print("Writting the following keys as input:")
    print([x["var_name"] for x in config["dataset"]["input_vars"]])

    # bin_idx = calc_bin_idx(input_data)

    # counts = onp.bincount(bin_idx)

    # sample_weights = 1/counts[bin_idx]
    if config["dataset"]["sample_weights"]["method"].lower() == "knn":
        sample_weights = compute_knn_weights(input_data)
    elif config["dataset"]["sample_weights"]["method"].lower() == "hist":
        bin_idx = calc_bin_idx(input_data)
        counts = onp.bincount(bin_idx)
        sample_weights = 1/counts[bin_idx]
    else: 
        print("No sample weights selected. Using uniform sample weights")
        sample_weights = onp.ones(len(df))[mask_exists&mask_cut]
        sample_weights /= onp.sum(sample_weights)

    #sample_weights = onp.array(df["fluxless_weight"])[mask_exists&mask_cut]

    sample_weights /= onp.sum(sample_weights)

    

    #sample_weights = torch.tensor(sample_weights)



    flux_vars = {}

    for flux_var in config["dataset"]["flux_vars"]:
        dtemp = onp.array(df[flux_var])[mask_exists&mask_cut]
        dtemp = dtemp
        flux_vars[flux_var] = dtemp

    # NNMFit needs true_energy
    if "MCPrimaryEnergy" in flux_vars:
        print("Renamed key 'MCPrimaryEnergy' into 'true_energy'")
        flux_vars["true_energy"] = flux_vars.pop("MCPrimaryEnergy")
    if "MCPrimaryDec" in flux_vars:
        print("Renamed key 'MCPrimaryDec' into 'true_dec'")
        flux_vars["true_dec"] = flux_vars.pop("MCPrimaryDec")
    if "MCPrimaryRA" in flux_vars:
        print("Renamed key 'MCPrimaryRA' into 'true_ra'")
        flux_vars["true_ra"] = flux_vars.pop("MCPrimaryRA")
    
    print(flux_vars.keys())



    if "additional_kwargs" in config["dataset"].keys():
        additional_kwargs = config["dataset"]["additional_kwargs"]
        kwargs_values={}
        for key in additional_kwargs:
            df_key = config["dataset"]["kwargs_values"][key]
            print(df_key," has the following number of entries ",len(onp.array(df[df_key])))
            kwargs_values[key] = onp.array(df[df_key])[mask_exists&mask_cut]

    else:
        additional_kwargs = ["none"]
        kwargs_values={"none":onp.ones(len(df))[mask_exists&mask_cut]}

    if weighter is not None:
        aux_jarr = {}
        for key in flux_vars.keys():
                aux_jarr[key] = Array(flux_vars[key])

        norm_weights = onp.array(weighter(aux_jarr))

    tot_norm_weight = onp.sum(norm_weights)

    #fluxless_weights = onp.array(df["fluxless_weight"])[mask_exists&mask_cut]
    ds = SimpleDataset(input_data,flux_vars,sample_weights,norm_weights,additional_kwargs,kwargs_values)


    return ds, sample_weights, tot_norm_weight"""


def make_jax_dataset(config,df,weighter=None):

    # TODO implement use of weighter to caclulate expected weights

    


    input_data, mask_exists, mask_cut = create_input_data(df,config["dataset"])
    input_data = input_data[mask_exists&mask_cut]
    print("NaNs in input data: ",jnp.isnan(input_data).sum())
    print("input data only contains finite values: ",jnp.isfinite(input_data).all())

    print("Writting the following keys as input:")
    print([x["var_name"] for x in config["dataset"]["input_vars"]])

    # bin_idx = calc_bin_idx(input_data)

    # counts = onp.bincount(bin_idx)

    # sample_weights = 1/counts[bin_idx]
    if config["dataset"]["sample_weights"]["method"].lower() == "knn":
        sample_weights = compute_knn_weights(input_data)
    elif config["dataset"]["sample_weights"]["method"].lower() == "hist":
        bin_idx = calc_bin_idx(input_data)
        counts = onp.bincount(bin_idx)
        sample_weights = 1/counts[bin_idx]
    else: 
        print("No sample weights selected. Using uniform sample weights")
        sample_weights = onp.ones(len(df))[mask_exists&mask_cut]
        sample_weights /= onp.sum(sample_weights)


    sample_weights /= onp.sum(sample_weights)
    sample_weights = Array(sample_weights)
    




    flux_vars = {}

    for flux_var in config["dataset"]["flux_vars"]:
        dtemp = onp.array(df[flux_var])[mask_exists&mask_cut]
        dtemp = Array(dtemp)
        flux_vars[flux_var] = dtemp

    # NNMFit needs true_energy
    if "MCPrimaryEnergy" in flux_vars:
        print("Renamed key 'MCPrimaryEnergy' into 'true_energy'")
        flux_vars["true_energy"] = flux_vars.pop("MCPrimaryEnergy")
    if "MCPrimaryDec" in flux_vars:
        print("Renamed key 'MCPrimaryDec' into 'true_dec'")
        flux_vars["true_dec"] = flux_vars.pop("MCPrimaryDec")
    if "MCPrimaryRA" in flux_vars:
        print("Renamed key 'MCPrimaryRA' into 'true_ra'")
        flux_vars["true_ra"] = flux_vars.pop("MCPrimaryRA")
    
    print(flux_vars.keys())



    if "additional_kwargs" in config["dataset"].keys():
        additional_kwargs = config["dataset"]["additional_kwargs"]
        kwargs_values={}
        for key in additional_kwargs:
            df_key = config["dataset"]["kwargs_values"][key]
            print(df_key," has the following number of entries ",len(Array(df[df_key])))
            kwargs_values[key] = Array(df[df_key])[mask_exists&mask_cut]

    else:
        additional_kwargs = ["none"]
        kwargs_values={"none":jnp.ones(len(df))[mask_exists&mask_cut]}


    #fluxless_weights = onp.array(df["fluxless_weight"])[mask_exists&mask_cut]
    input_data = Array(input_data)
    print("len data: ",len(input_data))
    for key in flux_vars.keys():
        print(f"len {key} flux var: ",len(flux_vars[key]))
    print("len sample weights: ",len(sample_weights))
    ds = input_data,flux_vars,sample_weights

    mask_tot = mask_exists&mask_cut

    return ds, sample_weights, mask_tot