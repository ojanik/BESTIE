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


def main(model_path,
         save_to=None,
         make_gif=False,
         make_weighted_hist=False,
         make_unweighted_hist=False,
         make_2D_scatter=False,
         make_trainstep_loss_curve=False,
         all_flag=False):

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

    batch_size = 10000
    num_parts = int(jnp.ceil(len(input_data)/batch_size))
    apply_fn = jit(net.apply)
    print("--- Calculating lss ---")
    for i in tqdm(range(num_parts),disable=True):
        if i == 0:
            lss = apply_fn({"params": params},input_data[i*batch_size:jnp.min(Array([(i+1)*batch_size,len(input_data)])),:len(config["dataset"]["input_vars"])])[:,0]
        else:
            lss = jnp.concatenate([lss,apply_fn({"params": params},input_data[i*batch_size:jnp.min(Array([(i+1)*batch_size,len(input_data)])),:len(config["dataset"]["input_vars"])])[:,0]])
    del(input_data)

    #shift lss to be between 0 and 1 like it is done during training
    mask = mask_exists&mask_cut
    mask2 = ~jnp.isnan(lss[mask])
    lss -= jnp.min(lss[mask][mask2])
    lss /= jnp.max(lss[mask][mask2])

    injected_params = config["injected_params"]
    obj = BESTIE.Optimization_Pipeline(config,list(injected_params.keys()))

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
        hvar = lss[mask]
        bins = onp.linspace(config["hists"]["bins_low"],config["hists"]["bins_up"],config["hists"]["bins_number"])
        H_total,_,_,_ = plt.hist2d(hvar,hvar,bins=[bins,bins],weights = obj.calc_weights(i_params,aux)[:,0])
        i_params["conv_norm"] = 0
        i_params["prompt_norm"] = 0
        H_astro,_,_,_ = plt.hist2d(hvar,hvar,bins=[bins,bins],weights = obj.calc_weights(i_params,aux)[:,0])
        i_params["astro_norm"] = 0
        i_params["conv_norm"] = 1.
        H_conv,_,_,_ = plt.hist2d(hvar,hvar,bins=[bins,bins],weights = obj.calc_weights(i_params,aux)[:,0])
        i_params["prompt_norm"] = 1.
        i_params["conv_norm"] = 0
        H_prompt,_,_,_ = plt.hist2d(hvar,hvar,bins=[bins,bins],weights = obj.calc_weights(i_params,aux)[:,0])



        plt.close()
        fig, ax = plt.subplots()
        plt.grid()
        #ax.stairs(H_total.sum(axis=1),jnp.linspace(-1,1,nob+1),label="total")
        ax.stairs(H_astro.sum(axis=1),bins,label="astro")
        ax.stairs(H_conv.sum(axis=1),bins,label="conv")
        ax.stairs(H_prompt.sum(axis=1),bins,label="prompt")
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
        hvar = lss[mask]
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

        plt.scatter(df["energy_truncated"],jnp.cos(Array(df["zenith_MPEFit"])),c=digi,cmap="tab20")
        plt.xscale("log")
        plt.xlabel("reco energy")
        plt.ylabel("cos(zenith)")
        plt.colorbar()

        plt.ylim(-1.05,0.05)
        plt.xlim(1e-1,1e9)

        plt.savefig(os.path.join(model_path,"2D_energy_coszenith.png"),dpi=256)
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


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Process some paths and an optional name.")
    
    # Add arguments
    parser.add_argument('--model_path', type=str, required=True, help="Path to saved model")

    parser.add_argument('--make_trainstep_loss_curve',action='store_true',help="Flag if loss curve for each training step should be created")
    parser.add_argument('--make_2D_scatter',action='store_true',help="Flag if 2D scatter plot with the colors corresponding to the bin should be created")
    parser.add_argument('--make_weighted_hist',action='store_true',help="Flag if flux weighted hist should be created")
    parser.add_argument('--make_unweighted_hist',action='store_true',help="Flag if unweighted hist should be created")

    parser.add_argument('--all',action='store_true',help="Enables all plotting options")

    parser.add_argument('--save_to',default=None,type=str,help="Optional path to ")



    parser.add_argument('--make_gif',action='store_true',help="Flag if gif of binning should be created")

    args = parser.parse_args()

    main(model_path=args.model_path,
         save_to=args.save_to,
         make_gif=args.make_gif,
         make_weighted_hist=args.make_weighted_hist,
         make_unweighted_hist=args.make_unweighted_hist,
         make_2D_scatter=args.make_2D_scatter,
         make_trainstep_loss_curve=args.make_trainstep_loss_curve,
         all_flag = args.all)
    
    print("DONE")
