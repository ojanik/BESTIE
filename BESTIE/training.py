import BESTIE
from BESTIE.data import SimpleDataset
from flax.training import train_state
import optax
from jax import random, jit, value_and_grad, nn
from jax.tree_util import tree_map
import jax.numpy as jnp
Array = jnp.array
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from functools import partial

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
    ds, sample_weights, tot_norm_weight = BESTIE.data.make_torch_dataset(config,df,weighter=weighter)
    
    tot_norm_weight = Array(tot_norm_weight)
    sampler = None
    shuffle = True
    drop_last = True

    # If True the dataset will be sampled to have the same number of samples in each bin
    # If False the dataset will be uniformly sampled
    if sample:
        from torch.utils.data import WeightedRandomSampler
        assert len(ds) == len(sample_weights)
        sampler = WeightedRandomSampler(weights=sample_weights,num_samples=config["training"]["batch_size"],replacement=False)

        shuffle = False
        drop_last = False

    batches_per_epoch = config["training"]["batches_per_epoch"]

    if config["training"]["average_gradients"]:
        update_steps_per_epoch = 1
    else:
        update_steps_per_epoch = batches_per_epoch

    dl = DataLoader(dataset=ds,
                    batch_size=config["training"]["batch_size"],
                    num_workers=0,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    sampler=sampler)


    

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
    init_params = obj.net.init(rng,jnp.ones(input_size))["params"]
    # Initialize the scale parameter to 0 which is needed for the number of bins
    # 0 corresponds to the number of bins which is set in the config
    init_params["scale"] = 0.
    init_params["Bscale"] = 0. #this is on a log scale


    result_dict["init_params"] = init_params

    
    # If True sample batches_per_epoch times the first batch of the dataloader
    if sample:
        it_dl = [list(dl)[0] for i in range(batches_per_epoch)]
    else:
        it_dl = dl

    # Set up learning rate scheduling
    lr_fn = BESTIE.nets.lr_handler(config,update_steps_per_epoch)

    # Set up optimizer for network parameters
    tx = getattr(optax,config["training"]["optimizer"].lower())(learning_rate = lr_fn)

    # Create state for network parameters
    state = train_state.TrainState.create(apply_fn=obj.net.apply,
                                          params=init_params,
                                          tx=tx)

    # Define learning rate scheduling for bin training TODO: also implement this in the config
    def lr_fn_bins(*args,**kwargs):
        return lr_fn(*args,**kwargs)*1e3

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


    # Build optimization pipeline
    pipe = obj.get_optimization_pipeline()

    # Create empty arrays to store the history of the training
    history = []
    history_steps = []
    history_losses = []
    lr_epochs = []
    number_of_bins = []

    running_losses_shape = len(config["loss"]["method"])

    # Start training loop
    # Loop over epochs
    for j in (tpbar:= tqdm(range(config["training"]["epochs"]))):
        # Reset running loss if average gradients are used
        running_loss = 0
        running_single_losses = Array(running_losses_shape*[0.])

        
        # Loop over batches
        pbar = tqdm(enumerate(it_dl), total=len(it_dl),disable=not trainstep_pbar)
        for i,(data,aux,sample_weights,norm_weights,kwargs) in pbar:
            data = Array(data)
            
            
            #sample_weights = (1-(1-Array(sample_weights))**config["training"]["batch_size"])/config["weights"]["upscale"] if sample else None
            sample_weights = Array(sample_weights)
            #norm_weights = jnp.squeeze(Array(norm_weights))
            

            sample_weights = Array(1/sample_weights / jnp.sum(1/sample_weights) * int(len(ds))) if sample else None
            for key in aux.keys():
                aux[key] = Array(aux[key])

            for key in kwargs.keys():
                kwargs[key] = Array(kwargs[key])

            @jit 
            def l(params):
                if config["dataset"]["fourier_feature_mapping"]["train_scale"]:
                    Bdata = BESTIE.data.fourier_feature_mapping.input_mapping(data,B,params["Bscale"])
                else: 
                    Bdata = BESTIE.data.fourier_feature_mapping.input_mapping(data,B,config["dataset"]["fourier_feature_mapping"]["scale"])
                loss, losses = pipe(params,
                                               injected_params=Array(list(injected_params.values())),
                                               data=Bdata,
                                               aux=aux,
                                               sample_weights=sample_weights,
                                               **kwargs)
                return loss, losses

            ((loss, losses), grads) = value_and_grad(l,has_aux=True)(state.params)
            
            

            if jnp.isnan(loss):
                raise ValueError("Loss is nan")

            if config["training"]["average_gradients"]:
                try:
                    collected_grads
                except:
                    collected_grads = BESTIE.utilities.jax_utils.scale_pytrees(0.,grads)
                collected_grads = BESTIE.utilities.jax_utils.add_pytrees(collected_grads,grads)
            else:
                grads_net_params = BESTIE.utilities.jax_utils.apply_mask(grads,train_mask)
                state = state.apply_gradients(grads=grads_net_params)
                grads_bins = BESTIE.utilities.jax_utils.apply_mask(grads,bins_mask)
                state_bins = state_bins.apply_gradients(grads=grads_bins)
                state.params["scale"] = state_bins.params["scale"]
                state.params["Bscale"] = state_bins.params["Bscale"]

            

            
            history_steps.append(loss)
            pbar.set_description(f"loss: {loss:,.6g}; losses: {[f'{l:,.2f}' for l in losses]} ; Bscale: {state.params['Bscale']:,.7f};number of bins: {config['hists']['bins_number']*2*nn.sigmoid(state.params['scale']):,.6g}")
            running_loss += loss
            running_single_losses += losses

        if config["training"]["average_gradients"]:


            average_grads = BESTIE.utilities.jax_utils.scale_pytrees(1/len(it_dl),collected_grads)

            grads_net_params = BESTIE.utilities.jax_utils.apply_mask(average_grads,train_mask)
            state = state.apply_gradients(grads=grads_net_params)
            grads_bins = BESTIE.utilities.jax_utils.apply_mask(average_grads,bins_mask)
            state_bins = state_bins.apply_gradients(grads=grads_bins)
            state.params["scale"] = state_bins.params["scale"]
            state.params["Bscale"] = state_bins.params["Bscale"]
            collected_grads = BESTIE.utilities.jax_utils.scale_pytrees(0.,collected_grads)

        lr = lr_fn(state.step)
        lr_epochs.append(lr)
        
        avg_loss = running_loss / batches_per_epoch
        avg_single_losses = running_single_losses / batches_per_epoch
        history.append(avg_loss)
        history_losses.append(avg_single_losses)
        number_of_bins.append(config['hists']['bins_number']*nn.sigmoid(state.params['scale']))

        tpbar.set_description(f"epoch loss: {avg_loss:.9f}")

        result_dict["history"] = history 
        result_dict["number_of_bins"] = number_of_bins
        result_dict["params"] = state.params
        result_dict["history_steps"] = history_steps
        result_dict["learning_rate_epochs"] = lr_epochs


        jnp.save(os.path.join(save_dir,"result.pickle"),result_dict,allow_pickle=True)


        with open(os.path.join(save_dir,"config.yaml"), 'w') as file:
            yaml.dump(config, file, default_flow_style=False)

        import matplotlib.pyplot as plt

        fig,ax = plt.subplots()
        plt.grid(True)

        ax2 = ax.twinx()

        ax.scatter(jnp.arange(len(history)),history,color="blue",label="loss")
        for i in range(running_losses_shape):
            ax.scatter(jnp.arange(len(history)),jnp.array(history_losses)[:,i],label=f"loss_{i}")
        ax2.scatter(jnp.arange(len(history)),number_of_bins,color="red",label="number of bins")
        ax2.set_ylabel("number of bins")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_yscale("log")
        plt.tight_layout()
        ax.legend()
        ax2.legend()
        plt.savefig(os.path.join(save_dir,"loss_curve.png"),dpi=256)
        plt.close()


    if plot_hists or plot_2D_scatter:
        from BESTIE.utilities import plot_routine

        plot_routine(model_path=save_dir,
                        make_unweighted_hist=plot_hists,
                        make_weighted_hist=plot_hists,
                        make_2D_scatter=plot_2D_scatter,
                        galactic=plot_galactic,
                        make_2D_scatter_galactic=plot_2D_scatter&plot_galactic,)

    run_inference = False

    if run_inference:
        print("-- Running inference for the whole dataset ---")
        raise NotImplementedError("Running inference is not implemented yet")
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
