import BESTIE
import jax.numpy as jnp
Array = jnp.array
import pandas as pd
import numpy as onp
from tqdm import tqdm
import matplotlib.pyplot as plt
from jax import jit, vmap, random

import os

import argparse


def plot_routine(model_path,
         save_to=None,
         make_gif=False,
         make_weighted_hist=False,
         make_unweighted_hist=False,
         make_2D_scatter=False,
         make_2D_scatter_galactic=False,
         make_trainstep_loss_curve=False,
         all_flag=False,
         galactic=False,
         save_df=False):

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
    del(input_data)

    kwargs["phi0"] = phi0[mask]

    transform_fun = BESTIE.transformations.transformation_handler(config["transformation"])

    lss = transform_fun(lss,**kwargs)

    #mask2 = ~jnp.isnan(lss[mask])
    #lss -= jnp.min(lss[mask][mask2])
    #lss /= jnp.max(lss[mask][mask2])

    injected_params = config["injected_params"]
    obj = BESTIE.Optimization_Pipeline(config,list(injected_params.keys()))

    print("lss: ",lss)
    print("lss_0: ",kwargs["lss0"])

    if make_trainstep_loss_curve or all_flag:
        print("--- Making trainstep loss curve ---")
        history = results_dict.item()["history_steps"]

        fig, ax = plt.subplots()
        ax.scatter(jnp.arange(len(history)),history)
        ax.set_yscale("log")
        ax.set_ylabel("loss")
        ax.set_xlabel("training step")
        plt.savefig(os.path.join(model_path,"trainstep_loss_curve.png"),dpi=256)
        plt.close()

    if make_weighted_hist or all_flag:
        print("--- Making weighted hist ---")
        i_params = injected_params.copy()
        hvar = lss#[mask]
        bins = onp.linspace(config["hists"]["bins_low"],config["hists"]["bins_up"],config["hists"]["bins_number"])
        H_total,_,_,_ = plt.hist2d(hvar,hvar,bins=[bins,bins],weights = obj.calc_weights(i_params,aux)[:,0])
        i_params["conv_norm"] = 0
        i_params["prompt_norm"] = 0
        if galactic:
            i_params["galactic_norm"] = 0
        H_astro,_,_,_ = plt.hist2d(hvar,hvar,bins=[bins,bins],weights = obj.calc_weights(i_params,aux)[:,0])
        i_params["astro_norm"] = 0
        i_params["conv_norm"] = 1.
        H_conv,_,_,_ = plt.hist2d(hvar,hvar,bins=[bins,bins],weights = obj.calc_weights(i_params,aux)[:,0])
        i_params["prompt_norm"] = 1.
        i_params["conv_norm"] = 0

        H_prompt,_,_,_ = plt.hist2d(hvar,hvar,bins=[bins,bins],weights = obj.calc_weights(i_params,aux)[:,0])

        if galactic:
            i_params["prompt_norm"] = 0.
            i_params["galactic_norm"] = 1.
            H_gal,_,_,_ = plt.hist2d(hvar,hvar,bins=[bins,bins],weights = obj.calc_weights(i_params,aux)[:,0])



        plt.close()
        fig, ax = plt.subplots()
        plt.grid()
        #ax.stairs(H_total.sum(axis=1),jnp.linspace(-1,1,nob+1),label="total")
        ax.stairs(H_astro.sum(axis=1),bins,label="astro")
        print(H_astro.sum(axis=1))
        ax.stairs(H_conv.sum(axis=1),bins,label="conv")
        ax.stairs(H_prompt.sum(axis=1),bins,label="prompt")
        if galactic:
            ax.stairs(H_gal.sum(axis=1),bins,label="galactic")
        ax.set_yscale("log")
        ax.set_xlabel("lss")
        ax.set_ylabel("events")
        ax.set_axisbelow(True)
        #ax.set_ylim(bottom=1e-6)
        #ax.set_xscale("log")
        plt.legend()
        plt.savefig(os.path.join(model_path,"weighted_hist.png"),dpi=256)
        plt.close()

    if make_unweighted_hist or all_flag:
        print("--- Making unweighted hist ---")
        hvar = lss#[mask]
        bins = onp.linspace(config["hists"]["bins_low"],config["hists"]["bins_up"],config["hists"]["bins_number"])
        H,_,_,_ = plt.hist2d(hvar,hvar,bins=[bins,bins])
        plt.close()

        fig, ax = plt.subplots()
        plt.grid()
        ax.stairs(H.sum(axis=1),bins)
        ax.set_yscale("log")
        ax.set_xlabel("lss")
        ax.set_ylabel("number of MC events, unweighted")

        plt.savefig(os.path.join(model_path,"unweighted_hist.png"),dpi=256)
        plt.close()

    if make_2D_scatter or all_flag:
        print("--- Making 2D scatter ---")
        nob = config["hists"]["bins_number"]
        digi = jnp.digitize(lss,bins=jnp.linspace(0,1,nob+1))

        plt.scatter(Array(df["energy_truncated"])[mask],jnp.cos(Array(df["zenith_MPEFit"]))[mask],c=digi,cmap="tab20")
        plt.xscale("log")
        plt.xlabel("reco energy")
        plt.ylabel("cos(reco zenith)")
        plt.colorbar()

        plt.ylim(-1.05,0.05)
        plt.xlim(1e1,1e8)

        plt.savefig(os.path.join(model_path,"2D_energy_coszenith.png"),dpi=256)
        plt.close()

    if make_2D_scatter_galactic or all_flag:
        print("--- Making 2D scatter ---")
        nob = config["hists"]["bins_number"]
        digi = jnp.digitize(lss,bins=jnp.linspace(0,1,nob+1))

        plt.scatter(Array(df["ra_MPEFit"])[mask],jnp.cos(Array(df["zenith_MPEFit"]))[mask],c=digi,cmap="tab20")
        #plt.xscale("log")
        plt.xlabel("reco ra")
        plt.ylabel("cos(reco zenith)")
        plt.colorbar()

        plt.ylim(-1.05,0.05)
        plt.xlim(-0.1,6.3)

        plt.savefig(os.path.join(model_path,"2D_ra_coszenith.png"),dpi=256)
        plt.close()

    if make_gif or all_flag:
        print("--- Making gif ---")
        import matplotlib.animation as animation

        nob = config["hists"]["bins_number"]
        bins = jnp.linspace(0, 1, nob + 1)
        digi = jnp.digitize(lss, bins=bins)

        fig, ax = plt.subplots()
        scatter = ax.scatter([], [], c=[], cmap="jet", vmin=0, vmax=1650)
        colorbar = fig.colorbar(scatter, ax=ax)

        def update(dd):
            ax.clear()
            scatter = ax.scatter(jnp.array(df["energy_truncated"])[digi == dd],
                                jnp.cos(jnp.array(df["zenith_MPEFit"]))[digi == dd],
                                c=digi[digi == dd],
                                cmap="jet", vmin=0, vmax=1650)
            ax.set_xscale("log")
            ax.set_xlabel("reco energy")
            ax.set_ylabel("cos(zenith)")
            ax.set_ylim(-1.05, 0.05)
            ax.set_xlim(1e-1, 1e9)
            return scatter,

        # Create the animation with a higher frame rate
        frame_rate = 30  # Adjust the frame rate as needed

        ani = animation.FuncAnimation(fig, update, frames=range(1, nob+1), blit=False, repeat=False)

        # Save the animation as a GIF
        ani.save(os.path.join(model_path,"2D_scatter.gif"), writer='pillow', fps=frame_rate)
        plt.close()

    if save_to is not None:
        df.insert(3,"lss",onp.array(lss))
        df.to_hdf(save_to,"a")
        print(f"Saved dataframe with lss at {save_to}")

    if save_df:

        print("--- Saving dataframe with updated lss values ---")

        assert len(df)==len(mask)
        df["lss"] = -1.
        df["lss"].mask(mask,lss,inplace=True)
        df.to_parquet()
        if ext[1:].lower() in ["parquet"]:
            df.to_parquet(dataframe)
        elif ext[1:].lower() in ["hdf","hd5"]:
            df.to_hdf(dataframe)
        
        print(f"--- Saved dataframe at {dataframe} ---")
