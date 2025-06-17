import BESTIE

from flax.training import train_state
import optax
from jax import random, jit, value_and_grad, nn
from jax.tree_util import tree_map
import jax.numpy as jnp
Array = jnp.array

from tqdm import tqdm
from functools import partial
import jax 
from jax.profiler import start_trace, stop_trace
import matplotlib.pyplot as plt
from jax import lax

import time

import pandas as pd

import yaml

import argparse
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def main(config,
        output_dir,
        name="unnamed",
        train_for_shape=False,
        sample=False,
        trainstep_pbar=False,
        plot_hists=False,
        plot_2D_scatter=False,
        plot_galactic=False):

    from datetime import datetime

    assert len(config["loss"]["signal_idx"]) == len(config["loss"]["weight_norm"])

    # Get the current date and time for file naming
    now = datetime.now()

    # Format the date and time as a string
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    save_dir = os.path.join(output_dir,name+"_"+date_time_str)
    config["save_dir"] = save_dir
    os.makedirs(save_dir, exist_ok=True)

    print(f"--- Results will be saved at {save_dir} ---")

    result_dict = {}

    injected_params = config["injected_params"].copy()
    for key in injected_params.keys():
        injected_params[key] = Array(injected_params[key])


    print(list(injected_params.keys()))

    

    print("--------------------- Loading and preparing data ---------------------")
    
    df = pd.read_parquet(config["dataset"]["dataframe"])
    
    # Save one entry of the dataframe which will be needed to build the weight graph
    df_one = df[:1]
    df_one.to_parquet(os.path.join(config["save_dir"],"df_one.parquet"))

    # Creating Pipeline Object
    config["dataset"]["length"] = int(len(df))
    obj = BESTIE.Optimization_Pipeline(config) #,list(injected_params.keys())
    weighter = partial(obj.calc_weights,injected_params)
    ds, sample_weights_all = BESTIE.data.make_jax_dataset(config,df,weighter=weighter)

    batches_per_epoch = config["training"]["batches_per_epoch"]

    if config["training"]["average_gradients"]:
        update_steps_per_epoch = 1
    else:
        update_steps_per_epoch = batches_per_epoch

    rng = random.key(config["rng"])
    

    # Creates the Fourier Feature Mapping
    B = BESTIE.data.fourier_feature_mapping.get_B(config["dataset"])
    if B is not None:
        input_size = 2 * int(config["dataset"]["fourier_feature_mapping"]["mapping_size"])
    else:
        input_size = len(config["dataset"]["input_vars"])

    print(f"--- The network has an input size of {input_size} ---")

    result_dict["ffm"] = {}
    result_dict["ffm"]["B"] = B

    print("--------------------- Initializing network ---------------------")
    # Initialize the network parameters
    if config["network"]["architecture"].lower() == "dense":
        init_params = obj.net.init(rng,jnp.ones(input_size))["params"]
    elif config["network"]["architecture"].lower() == "transformer":
        init_params = obj.net.init(rng,(jnp.ones((1,config["training"]["batch_size"],input_size))))["params"]
    # Initialize the scale parameter to 0 which is needed for the number of bins
    # 0 corresponds to the number of bins which is set in the config
    init_params["scale"] = 0.
    init_params["Bscale"] = config["dataset"]["fourier_feature_mapping"]["scale"] #this is on a log scale


    result_dict["init_params"] = init_params

    # Set up learning rate scheduling
    lr_fn = BESTIE.nets.lr_handler(config,update_steps_per_epoch)

    # Set up optimizer for network parameters
    tx = getattr(optax,config["training"]["optimizer"].lower())(learning_rate = lr_fn)

    # Create state for network parameters
    state = train_state.TrainState.create(apply_fn=obj.net.apply,
                                          params=init_params,
                                          tx=tx)

    def count_params(params):
        sizes = jax.tree_util.tree_map(lambda x: jnp.size(x), params)
        return sum(jax.tree_util.tree_leaves(sizes))

    num_params = count_params(state.params)
    print(f"🧠 Total number of parameters: {num_params}")
    
    # Define learning rate scheduling for bin training TODO: also implement this in the config
    def lr_fn_bins(*args,**kwargs):
        return lr_fn(*args,**kwargs)

    # Set up optimizer for bin parameters
    tx_bins = getattr(optax,config["training"]["optimizer"].lower())(learning_rate = lr_fn_bins) 

    # Setup state for bin parameters
    # Could this be combined with the state for the network parameters? The problem is the different learning rates
    state_bins = train_state.TrainState.create(apply_fn=obj.net.apply,
                                          params=init_params,
                                          tx=tx_bins)
    
    # Create masks for the network and bin parameters
    train_mask = tree_map(lambda _: jnp.ones_like(_), init_params)
    train_mask["scale"] = False
    train_mask["Bscale"] = False

    bins_mask = tree_map(lambda _: jnp.zeros_like(_),init_params)
    assert isinstance(config["training"]["train_number_of_bins"],bool)
    bins_mask["scale"] = config["training"]["train_number_of_bins"]
    bins_mask["Bscale"] = True

    # Save the config
    with open(os.path.join(save_dir,"config.yaml"), 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    # Build optimization pipeline
    pipe = obj.get_optimization_pipeline()


    # Define the loss function
    @jit 
    def l(params,batch):
        data, aux, sample_weights = batch
        if config["dataset"]["fourier_feature_mapping"]["train_scale"]:
            Bdata = BESTIE.data.fourier_feature_mapping.input_mapping(data,B,params["Bscale"])
        else: 
            Bdata = BESTIE.data.fourier_feature_mapping.input_mapping(data,B,config["dataset"]["fourier_feature_mapping"]["scale"])
        loss, losses = pipe(params,
                                        injected_params=Array(list(injected_params.values())),
                                        data=Bdata,
                                        aux=aux,
                                        sample_weights=sample_weights
                                        )
        return loss, losses

    @jit 
    def train_epoch(state, state_bins, rng):
        # Loop over batches
        data, aux, sample_weights = ds

        def step_fn(carry,batch_idx):
            state, state_bins, rng, accum_grads = carry
            rng, subkey = random.split(rng)
            
            indices = jax.random.choice(subkey, data.shape[0], shape=(config["training"]["batch_size"],),p=sample_weights_all, replace=False)


            batch = data[indices], jax.tree_util.tree_map(lambda v: v[indices], aux), Array(1/sample_weights[indices] / jnp.sum(1/sample_weights[indices]) * data.shape[0])
            #Compute grads
            (loss, losses), grads = jax.value_and_grad(l, has_aux=True)(state.params, batch)

            if config["training"]["average_gradients"]:
                accum_grads = BESTIE.utilities.jax_utils.add_pytrees(accum_grads, grads)  # Accumulate
            else:
                # Apply mask and update
                grads_net = BESTIE.utilities.jax_utils.apply_mask(grads, train_mask)
                grads_bins = BESTIE.utilities.jax_utils.apply_mask(grads, bins_mask)
                state = state.apply_gradients(grads=grads_net)
                state_bins = state_bins.apply_gradients(grads=grads_bins)
                new_params = {**state.params, "scale": state_bins.params["scale"], "Bscale": state_bins.params["Bscale"]}
                state = state.replace(params=new_params)
                accum_grads = jax.tree_util.tree_map(jnp.zeros_like, grads)  # No accumulation

            carry = (state, state_bins, rng, accum_grads)
            metrics = (loss, losses)
            return carry, metrics

        # Init
        rng, init_key = jax.random.split(rng)
        accum_grads = BESTIE.utilities.jax_utils.scale_pytrees(0., state.params)
        (state, state_bins, _, accum_grads), metrics = lax.scan(
            step_fn, (state, state_bins, init_key, accum_grads), jnp.arange(config["training"]["batches_per_epoch"])
        )

        if config["training"]["average_gradients"]:
            # Apply accumulated gradients
            grads_net = BESTIE.utilities.jax_utils.apply_mask(accum_grads, train_mask)
            grads_bins = BESTIE.utilities.jax_utils.apply_mask(accum_grads, bins_mask)
            state = state.apply_gradients(grads=grads_net)
            state_bins = state_bins.apply_gradients(grads=grads_bins)
            new_params = {**state.params, "scale": state_bins.params["scale"], "Bscale": state_bins.params["Bscale"]}
            state = state.replace(params=new_params)
        return state, state_bins, metrics

    history = []
    history_losses = []
    lr_epochs = []
    number_of_bins = []
    Bscales = []

    #for i in range(config["training"]["epochs"]):
    # --- Compilation timing ---
    rng, subkey = random.split(rng)
    compile_start = time.time()
    state, state_bins, metrics = train_epoch(state, state_bins, subkey)
    jax.block_until_ready(metrics[0])  # Force JIT to complete
    loss, losses  = metrics
    if jnp.isnan(jnp.mean(loss)):
        raise ValueError("Loss has become nan")
    print(f"Epoch 0: loss: {jnp.mean(loss):.6g}; Bscale: {state.params['Bscale']:,.7f}; number of bins: {config['hists']['bins_number']*2*nn.sigmoid(state.params['scale']):,.6g}")
    compile_time = time.time() - compile_start
    print(f"⏱️ Compilation time: {compile_time:.2f} seconds")

    # --- Execution timing ---
    start_time = time.time()
    for i in range(1,config["training"]["epochs"]):
        rng, subkey = random.split(rng)
        state, state_bins, metrics = train_epoch(state, state_bins, subkey)
        loss, losses = metrics
        history.append(jnp.mean(loss))
        history_losses.append(losses)
        Bscales.append(state.params['Bscale'])
        number_of_bins.append(config['hists']['bins_number'] * 2 * nn.sigmoid(state.params['scale']))
        print(f"Epoch {i}: loss: {jnp.mean(loss):.6g}; Bscale: {state.params['Bscale']:,.7f}; number of bins: {config['hists']['bins_number']*2*nn.sigmoid(state.params['scale']):,.6g}")
        if i % 100 == 0:
            jax.block_until_ready(metrics[0])  # Force JIT to complete
            plots = []
            fig, ax = plt.subplots()
            plt.grid(True)
            p1 = ax.scatter(jnp.arange(len(history)), history, color="blue", label="loss",s=1)
            plots.append(p1)
            ax.set_ylabel("loss")
            if config["training"]["train_number_of_bins"]:
                ax2 = ax.twinx()
                p2 = ax2.scatter(jnp.arange(len(history)), number_of_bins, color="red", label="number of bins",s=1)
                plots.append(p2)
                ax2.set_ylabel("number of bins")
            if config["dataset"]["fourier_feature_mapping"]["train_scale"]:
                ax3 = ax.twinx()
                if config["training"]["train_number_of_bins"]:
                    # Offset axis if this is the third
                    ax3.spines.right.set_position(("axes", 1.2))
                p3 = ax3.scatter(jnp.arange(len(history)),Bscales,color="green",label="log ffm scale",s=1)
                plots.append(p3)
                ax3.set_ylabel("log fmm scale")
            ax.set_yscale("log")
                    
            ax.set_xlabel("epoch")
            plt.tight_layout()
            ax.legend(handles=plots)
            plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=256)
            plt.close()

    end_time = time.time()

    print(f"🚀 Training time: {end_time - start_time:.2f} seconds")

    
    plots = []
    fig, ax = plt.subplots()
    plt.grid(True)
    p1 = ax.scatter(jnp.arange(len(history)), history, color="blue", label="loss",s=1)
    plots.append(p1)
    ax.set_ylabel("loss")
    if config["training"]["train_number_of_bins"]:
        ax2 = ax.twinx()
        p2 = ax2.scatter(jnp.arange(len(history)), number_of_bins, color="red", label="number of bins",s=1)
        plots.append(p2)
        ax2.set_ylabel("number of bins")
    if config["dataset"]["fourier_feature_mapping"]["train_scale"]:
        ax3 = ax.twinx()
        if config["training"]["train_number_of_bins"]:
            # Offset axis if this is the third
            ax3.spines.right.set_position(("axes", 1.2))
        p3 = ax3.scatter(jnp.arange(len(history)),Bscales,color="green",label="log ffm scale",s=1)
        plots.append(p3)
        ax3.set_ylabel("log ffm scale")
    ax.set_yscale("log")
            
    ax.set_xlabel("epoch")
    plt.tight_layout()
    ax.legend(handles=plots)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=256)
    plt.close()


    result_dict["history"] = history 
    result_dict["number_of_bins"] = number_of_bins
    result_dict["params"] = state.params
    result_dict["learning_rate_epochs"] = lr_epochs


    jnp.save(os.path.join(save_dir,"result.pickle"),result_dict,allow_pickle=True)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some paths and an optional name.")
    
    # Add arguments
    parser.add_argument('--config_path', type=str, help="Path to the config file")
    parser.add_argument('--dataset_config_path', type=str, help="Path to the dataset config file")
    parser.add_argument('--output_dir', type=str, help="Path to the output directory")
    parser.add_argument('--name', type=str, help="Optional name")
    parser.add_argument('--train_for_shape',action='store_true',help="If shape should be trained")
    parser.add_argument('--sample',action='store_true',help="If shape should be sampled")
    parser.add_argument('--trainstep_pbar',action='store_true',help="Flag to enable a progress bar for each epoch")
    parser.add_argument('--overrides',nargs='+',default=[])
    parser.add_argument('--plot_hists',action='store_true',default=False,help="Flag if unweighted and weighted hists should be created")
    parser.add_argument('--plot_2D_scatter',action="store_true",default=False,help="Flag if 2D scatter plot in energy and cos zenith should be done")
    parser.add_argument('--plot_galactic',action='store_true',default=False,help="If plot_hists is also set, then the galactic component is included, if plot_2D_scatter then also 2D_scatter in cos zenith and ra is plotted")
    args = parser.parse_args()



    config = BESTIE.utilities.configs.parse_yaml(args.config_path)

    config["dataset"] = BESTIE.utilities.configs.parse_yaml(args.dataset_config_path)

    for override in args.overrides:
        override_dict = BESTIE.utilities.configs.parse_yaml(override)
        config = BESTIE.utilities.configs.override(config,override_dict)

    main(config=config,
            output_dir=args.output_dir, 
            name=args.name,
            train_for_shape=args.train_for_shape,
            sample=args.sample,
            trainstep_pbar=args.trainstep_pbar,
            plot_hists=args.plot_hists,
            plot_2D_scatter=args.plot_2D_scatter,
            plot_galactic=args.plot_galactic)
        
    print("DONE")
