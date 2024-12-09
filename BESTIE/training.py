import BESTIE
from BESTIE.data import SimpleDataset
from flax.training import train_state
import optax
from jax import random, jit, value_and_grad, nn, tree_map
import jax.numpy as jnp
Array = jnp.array
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from functools import partial

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

    

    # Get the current date and time
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
    # Creating Pipeline Object
    

    print("--------------------- Loading and preparing data ---------------------")
    # function to calculate the weights 
    #weighter = partial(obj.calc_weights,injected_params)
    ds, sample_weights = BESTIE.data.make_torch_dataset(config,weighter=None)
    obj = BESTIE.Optimization_Pipeline(config,list(injected_params.keys()))
    sampler = None
    shuffle = True
    drop_last = True

    if sample:
        from torch.utils.data import WeightedRandomSampler
        #sample_weights = torch.load(config["sample_weights_path"])
        sampler = WeightedRandomSampler(weights=sample_weights,num_samples=config["training"]["batch_size"],replacement=False)

        assert len(ds) == len(sample_weights)

        shuffle = False
        drop_last = False
    batches_per_epoch = config["training"]["batches_per_epoch"]#int(jnp.floor(len(ds)/config["training"]["batch_size"]))

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


    #config["weights"]["upscale"] = len(ds)/config["training"]["batch_size"]

    

    rng = random.key(config["rng"])
    
    input_size = len(config["dataset"]["input_vars"])
    B = BESTIE.data.fourier_feature_mapping.get_B(config["dataset"])
    if B is not None:
        input_size = 2 * int(config["dataset"]["fourier_feature_mapping"]["mapping_size"])

    print(f"--- The network has an input size of {input_size} ---")

    result_dict["ffm"] = {}
    result_dict["ffm"]["B"] = B
    init_params = obj.net.init(rng,jnp.ones(input_size))["params"]
    
    init_params["scale"] = 0.

    result_dict["init_params"] = init_params

    it_dl = dl

    if sample:
        it_dl = [list(dl)[0] for i in range(batches_per_epoch)]

    if train_for_shape:
        print("--- Training for shape --")
        init_params = BESTIE.train_shape(obj.net,init_params,it_dl,config)
        result_dict["shape_params"] = init_params
        jnp.save(os.path.join(save_dir,"result.pickle"),result_dict,allow_pickle=True)

    ########################################################################################################################################

    print(100*"-")
    #setup train state
    lr_fn = BESTIE.nets.lr_handler(config,update_steps_per_epoch)

    tx = getattr(optax,config["training"]["optimizer"].lower())(learning_rate = lr_fn)

    state = train_state.TrainState.create(apply_fn=obj.net.apply,
                                          params=init_params,
                                          tx=tx)

    state_bins = train_state.TrainState.create(apply_fn=obj.net.apply,
                                          params=init_params,
                                          tx=tx)

    #state_bins = train_state.TrainState.create(apply_fn=obj.net.apply,
    #                                           params=init_params,
    #                                           tx=tx)
    
    train_mask = tree_map(lambda _: jnp.ones_like(_), init_params)
    train_mask["scale"] = 0

    bins_mask = tree_map(lambda _: jnp.zeros_like(_),init_params)
    bins_mask["scale"] = config["training"]["train_number_of_bins"]



    #net_params_mask = tree_map(lambda _: 1 - _, bin_params_mask)


    pipe = obj.get_optimization_pipeline()
    #pipe = jit(pipe)

    

    #training loop

    """print("--- Start data hist test ---")

    import time
    # Start time
    start_time = time.time()

    # Run the function
    dl = DataLoader(dataset=ds,
                    batch_size=len(ds),
                    num_workers=0,
                    shuffle=shuffle,
                    drop_last=drop_last)
    
    input_data, aux, sample_weights, kwargs = next(iter(dl_))
    input_data = Array(input_data)
    for key in aux.keys():
                aux[key] = Array(aux[key])
    input_data = BESTIE.data.fourier_feature_mapping.input_mapping(input_data,B)
    obj.calc_data_hist(init_params,input_data,aux,Array(list(injected_params.values())))
    print(obj.data_hist)

    # End time
    end_time = time.time()

    # Time taken
    execution_time = end_time - start_time
    print(f"Time taken by the function: {execution_time} seconds")
    print(f"Time taken by the function: {execution_time} seconds")

    print("--- Test done ---")
    quit()"""

    history = []
    history_steps = []
    lr_epochs = []
    number_of_bins = []

    #total_weight = obj.calc_weights(injected_params,)

    for j in (tpbar:= tqdm(range(config["training"]["epochs"]))):
        running_loss = 0

        

        pbar = tqdm(enumerate(it_dl), total=len(it_dl),disable=not trainstep_pbar)
        for i,(data,aux,sample_weights,kwargs) in pbar:
            data = Array(data)
            
            data = BESTIE.data.fourier_feature_mapping.input_mapping(data,B)
            #sample_weights = (1-(1-Array(sample_weights))**config["training"]["batch_size"])/config["weights"]["upscale"] if sample else None
            sample_weights = Array(sample_weights) if sample else None
            for key in aux.keys():
                aux[key] = Array(aux[key])

            for key in kwargs.keys():
                kwargs[key] = Array(kwargs[key])

            kwargs["phi0"] = obj.net.apply({"params":init_params},data)[:,0]

            #print(kwargs)
            #quit()

            """lss = obj.get_lss(state.params,data)

            hist = obj.get_hist(lss,Array(list(injected_params.values())),aux,sample_weights=sample_weights)

            print(hist.sum())
            continue"""

            #ap = obj.get_analysis_pipeline()
            #llh = ap(Array(list(injected_params.values())),lss,aux,hist)
            #print(llh)

            loss, grads = value_and_grad(pipe)(state.params,
                                               injected_params=Array(list(injected_params.values())),
                                               data=data,
                                               aux=aux,
                                               sample_weights=sample_weights,
                                               **kwargs)
            
            

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

            

            
            history_steps.append(loss)
            pbar.set_description(f"loss: {loss:,.6g} ; number of bins: {config['hists']['bins_number']*nn.sigmoid(state.params['scale']):,.6g}")
            running_loss += loss

        if config["training"]["average_gradients"]:


            average_grads = BESTIE.utilities.jax_utils.scale_pytrees(1/len(it_dl),collected_grads)
            #print(average_grads)

            grads_net_params = BESTIE.utilities.jax_utils.apply_mask(average_grads,train_mask)
            state = state.apply_gradients(grads=grads_net_params)
            grads_bins = BESTIE.utilities.jax_utils.apply_mask(average_grads,bins_mask)
            state_bins = state_bins.apply_gradients(grads=grads_bins)

            collected_grads = BESTIE.utilities.jax_utils.scale_pytrees(0.,collected_grads)

        lr = lr_fn(state.step)
        lr_epochs.append(lr)
        
        avg_loss = running_loss / batches_per_epoch
        history.append(avg_loss)
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
        ax2.scatter(jnp.arange(len(history)),number_of_bins,color="red",label="number of bins")
        ax2.set_ylabel("number of bins")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_yscale("log")
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(save_dir,"loss_curve.png"),dpi=256)
        plt.close()

        #jax.profiler.save_device_memory_profile("memory.prof")    

    if plot_hists or plot_2D_scatter:
        from BESTIE.utilities import plot_routine

        plot_routine(model_path=save_dir,
                        make_unweighted_hist=plot_hists,
                        make_weighted_hist=plot_hists,
                        make_2D_scatter=plot_2D_scatter,
                        galactic=plot_galactic,
                        make_2D_scatter_galactic=plot_2D_scatter&plot_galactic,
                        galactic_contour_path="/home/saturn/capn/capn105h/data/cringe_contour.npz")

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


    """wide_layer = {"layer":"Dense","size":1650,"activation":"relu"}

    for i in range(5):
        
        
        name = f"MSU_bkde_lin_{i}_wide_layers"
        print(f"Now training {name}")
        main(config=config, 
            output_dir=args.output_dir, 
            name=name,
            train_for_shape=args.train_for_shape,
            sample=args.sample,
            no_trainstep_pbar=args.no_trainstep_pbar)
        
        config["network"]["hidden_layers"].insert(len(config["network"]["hidden_layers"])-1,wide_layer)"""
        
    print("DONE")
