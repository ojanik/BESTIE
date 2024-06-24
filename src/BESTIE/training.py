import BESTIE
from BESTIE.data_loaders import SimpleDataset
from flax.training import train_state
import optax
from jax import random, jit, value_and_grad
import jax.numpy as jnp
Array = jnp.array
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import argparse
import os

import jax

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def main(config_path,output_dir,name="unnamed",train_for_shape=False):
    
    config = BESTIE.utilities.parse_yaml(config_path)

    from datetime import datetime

    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    save_dir = os.path.join(output_dir,name+"_"+date_time_str)

    os.makedirs(save_dir, exist_ok=True)

    print(f"--- Results will be saved at {save_dir} ---")

    result_dict = {}

    injected_params = {"astro_norm":Array(1.36),
                   "gamma_astro":Array(2.37),
                   "prompt_norm":Array(1.),
                   "conv_norm":Array(1.),
                   "CR_grad":Array(0.),
                   "delta_gamma":Array(0.)}


    #get dataset and dataloader
    ds = torch.load(config["dataset_path"])
    dl = DataLoader(ds,batch_size=config["training"]["batch_size"],num_workers=0,shuffle=True,drop_last=True)

    config["weights"]["upscale"] = len(dl)

    #setup train state

    obj = BESTIE.Optimization_Pipeline(config,list(injected_params.keys()))

    rng = random.key(config["rng"])
    init_params = obj.net.init(rng,jnp.ones(config["network"]["input_size"]))["params"]

    if train_for_shape:
        print("--- Training for shape --")
        init_params = BESTIE.train_shape(obj.net,init_params,ds,config)
        result_dict["shape_params"] = init_params
        jnp.save(os.path.join(save_dir,"result.pickle"),result_dict,allow_pickle=True)

    print(100*"-")

    steps_per_epoch = len(dl)
    lr = BESTIE.nets.lr_handler(config,steps_per_epoch)

    tx = getattr(optax,config["training"]["optimizer"].lower())(learning_rate = lr)

    state = train_state.TrainState.create(apply_fn=obj.net.apply,
                                          params=init_params,
                                          tx=tx)

    pipe = obj.get_optimization_pipeline()
    pipe = jit(pipe)

    

    #training loop

    history = []
    history_steps = []
    for j in (tpbar:= tqdm(range(config["training"]["epochs"]))):
        running_loss = 0
        pbar = tqdm(enumerate(dl), total=len(dl))
        for i,(data,aux) in pbar:
            data = Array(data)

            for key in aux.keys():
                aux[key] = Array(aux[key])

            #data_hist = asimov_func(state.params,Array(list(injected_params.values())),data,aux)
            for k in tqdm(range(50)):
                loss, grads = value_and_grad(pipe)(state.params,Array(list(injected_params.values())),data,aux)
                state = state.apply_gradients(grads=grads)
                
            history_steps.append(loss)
            pbar.set_description(f"loss: {loss:.9f}")
            running_loss += loss
        
        avg_loss = running_loss / len(dl)
        history.append(avg_loss)

        tpbar.set_description(f"epoch loss: {avg_loss:.9f}")

    result_dict = {}
    result_dict["history"] = history 
    result_dict["params"] = state.params
    result_dict["history_steps"] = history_steps

    from datetime import datetime

    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    save_dir = os.path.join(output_dir,name+"_"+date_time_str)

    os.makedirs(save_dir, exist_ok=True)

    jnp.save(os.path.join(save_dir,"result.pickle"),result_dict,allow_pickle=True)

    import yaml

    with open(os.path.join(save_dir,"config.yaml"), 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    import matplotlib.pyplot as plt

    fig,ax = plt.subplots()
    ax.scatter(jnp.arange(len(history)),history)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir,"loss_curve.png"),dpi=256)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some paths and an optional name.")
    
    # Add arguments
    parser.add_argument('--config_path', type=str, help="Path to the config file")
    parser.add_argument('--output_dir', type=str, help="Path to the output directory")
    parser.add_argument('--name', type=str, help="Optional name")

    args = parser.parse_args()

    main(config_path=args.config_path, output_dir=args.output_dir, name=args.name)