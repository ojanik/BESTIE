import BESTIE
import jax.numpy as jnp
Array = jnp.array
import os
from jax import jit, nn
from tqdm import tqdm
import numpy as onp
import pandas as pd

def calc_lss(model_path):
    path_to_config = os.path.join(model_path,"config.yaml")
    config = BESTIE.utilities.parse_yaml(path_to_config)

    model = BESTIE.nets.model_handler(config)
    net = model()
    dataframe = config["dataset"]["dataframe"]
    _,ext = os.path.splitext(dataframe) #get the correct extension for reading in the dataframe (probably either hdf or parquet)
    print("--- reading dataframe ---")
    if ext[1:].lower() in ["parquet"]:
        df = pd.read_parquet(dataframe)
    elif ext[1:].lower() in ["hdf","hd5"]:
        df = pd.read_hdf(dataframe)
    print("--- creating input data and masks ---")
    print("--- ignore warnings about dividing by zero, those events are filtered out in the end ---")
    input_data, mask_exists, mask_cut = BESTIE.data.prepare_data.create_input_data(df,config["dataset"])

    aux = {}
    for flux_var in config["dataset"]["flux_vars"]:
        dtemp = onp.array(df[flux_var])
        dtemp = dtemp[mask_exists&mask_cut]
        aux[flux_var] = dtemp

    # NNMFit needs true_energy
    if "MCPrimaryEnergy" in aux:
        print("Renamed key 'MCPrimaryEnergy' into 'true_energy'")
        aux["true_energy"] = aux.pop("MCPrimaryEnergy")

    results_dict = jnp.load(os.path.join(model_path,"result.pickle.npy"),allow_pickle=True)
    params = results_dict.item()["params"]

    B = results_dict.item()["ffm"]["B"]

    batch_size = 10000
    num_parts = int(jnp.ceil(len(input_data)/batch_size))
    apply_fn = jit(net.apply)
    print("--- Calculating lss ---")
    for i in tqdm(range(num_parts),disable=True):
        batched_input_data = input_data[i*batch_size:jnp.min(Array([(i+1)*batch_size,len(input_data)])),:len(config["dataset"]["input_vars"])]
        batched_input_data = BESTIE.data.fourier_feature_mapping.input_mapping(batched_input_data,B)
        if i == 0:
            lss = apply_fn({"params": params},batched_input_data)[:,0]
        else:
            lss = jnp.concatenate([lss,apply_fn({"params": params},batched_input_data)[:,0]])
    

    #shift lss to be between 0 and 1 like it is done during training
    # transform function

    
    
    mask = mask_exists&mask_cut

    lss = lss[mask]

    kwargs = {}
    kwargs["lss0"] = Array(df["lss0_standard_binning"])[mask]
    
    #phi0

    init_params = results_dict.item()["init_params"]
    for i in tqdm(range(num_parts),disable=True):
        batched_input_data = input_data[i*batch_size:jnp.min(Array([(i+1)*batch_size,len(input_data)])),:len(config["dataset"]["input_vars"])]
        batched_input_data = BESTIE.data.fourier_feature_mapping.input_mapping(batched_input_data,B)
        if i == 0:
            phi0 = apply_fn({"params": init_params},batched_input_data)[:,0]
        else:
            phi0 = jnp.concatenate([phi0,apply_fn({"params": init_params},batched_input_data)[:,0]])
    #del(input_data)

    kwargs["phi0"] = phi0[mask]

    transform_fun = BESTIE.transformations.transformation_handler(config["transformation"])

    lss = transform_fun(lss,**kwargs)
    
    bin_scale_up = config["hists"]["bins_up"]*nn.sigmoid(params["scale"]) * 2
    
    lss *= bin_scale_up

    return lss


def calc_lss_for_df(df,model_path,disable_pbar=True,oversample_shift = 0.):
    path_to_config = os.path.join(model_path,"config.yaml")
    config = BESTIE.utilities.parse_yaml(path_to_config)

    model = BESTIE.nets.model_handler(config)
    net = model()
    
    print("--- creating input data and masks ---")
    print("--- ignore warnings about dividing by zero, those events are filtered out in the end ---")
    input_data, mask_exists, mask_cut = BESTIE.data.prepare_data.create_input_data(df,config["dataset"])

    aux = {}
    for flux_var in config["dataset"]["flux_vars"]:
        dtemp = onp.array(df[flux_var])
        dtemp = dtemp[mask_exists&mask_cut]
        aux[flux_var] = dtemp

    # NNMFit needs true_energy
    if "MCPrimaryEnergy" in aux:
        print("Renamed key 'MCPrimaryEnergy' into 'true_energy'")
        aux["true_energy"] = aux.pop("MCPrimaryEnergy")

    results_dict = jnp.load(os.path.join(model_path,"result.pickle.npy"),allow_pickle=True)
    params = results_dict.item()["params"]

    B = results_dict.item()["ffm"]["B"]

    batch_size = 10000
    num_parts = int(jnp.ceil(len(input_data)/batch_size))
    apply_fn = jit(net.apply)
    print("--- Calculating lss ---")
    for i in tqdm(range(num_parts),disable=disable_pbar):
        batched_input_data = input_data[i*batch_size:jnp.min(Array([(i+1)*batch_size,len(input_data)])),:len(config["dataset"]["input_vars"])]
        batched_input_data = BESTIE.data.fourier_feature_mapping.input_mapping(batched_input_data,B)
        if i == 0:
            lss = apply_fn({"params": params},batched_input_data)[:,0]
        else:
            lss = jnp.concatenate([lss,apply_fn({"params": params},batched_input_data)[:,0]])
    

    #shift lss to be between 0 and 1 like it is done during training
    # transform function

    
    
    mask = mask_exists&mask_cut

    lss = lss[mask]

    kwargs = {}
    kwargs["lss0"] = Array(df["lss0_standard_binning"])[mask]
    
    #phi0

    init_params = results_dict.item()["init_params"]
    for i in tqdm(range(num_parts),disable=True):
        batched_input_data = input_data[i*batch_size:jnp.min(Array([(i+1)*batch_size,len(input_data)])),:len(config["dataset"]["input_vars"])]
        batched_input_data = BESTIE.data.fourier_feature_mapping.input_mapping(batched_input_data,B)
        if i == 0:
            phi0 = apply_fn({"params": init_params},batched_input_data)[:,0]
        else:
            phi0 = jnp.concatenate([phi0,apply_fn({"params": init_params},batched_input_data)[:,0]])
    #del(input_data)

    kwargs["phi0"] = phi0[mask]

    transform_fun = BESTIE.transformations.transformation_handler(config["transformation"])

    lss = transform_fun(lss,**kwargs)
    
    bin_scale_up = config["hists"]["bins_up"]*nn.sigmoid(params["scale"]) * 2
    
    lss *= bin_scale_up

    return lss
