from datetime import datetime

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

from pathlib import Path

from ..pipeline import Optimization_Pipeline
from .. import utilities, nets
from .. import data as BESTIEdata

class Train(Optimization_Pipeline):

    def __init__(self,config=None,result_dir=None,name="unnamed"):
        if result_dir is not None:
            self.result_dir = result_dir
            print("--- Loading from result directory ---")
            print("Given config will be ignored")
            self.load_from_result_dir()
            
        else:
            self.result_dict = None
            if config is None:
                raise ValueError("No config provided")
            self.config = config
            self.set_result_dict()
            self._make_result_dir(name)
          
        self.load_data()
        super().__init__(self.config)
        
        self.build_dataset()


        self.rng = random.key(self.config["rng"])
        self._set_optimization_pipeline()
        self.initialize_network()
        self.train_val_test_split()
        self.train_epoch = self.build_train_step(ds=self.ds_train,weights=self.weights_train,training=True)
        self.val_epoch = self.build_train_step(ds=self.ds_val,weights=self.weights_val,training=False)
        data, aux, sample_weights = self.ds_train
        self.train_data_shape = jnp.shape(data)
        self.train_data_shape = tuple(map(int, self.train_data_shape))
        print("Train data shape: ",self.train_data_shape)

        #print(data.shape)
        #quit()
        self.save_config()

    def load_from_result_dir(self):
        self.config = utilities.configs.parse_yaml(os.path.join(self.result_dir,"config.yaml"))
        self.result_dict = jnp.load(os.path.join(self.result_dir,"result.pickle.npy"),allow_pickle=True).item()
        self.B = self.result_dict["ffm"]["B"]
        
    def _make_result_dir(self,name="unnamed"):

        if not "save_dir" in self.config:
            # Get current time for 
            now = datetime.now()
            # Format the date and time as a string
            date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
            save_dir = os.path.join(self.config["output_dir"],name+"_"+date_time_str)
            self.config["save_dir"] = save_dir
            os.makedirs(self.config["save_dir"], exist_ok=True)
        
            print(f"--- Results will be saved at {self.config['save_dir']} ---")

        else:
            print(f"--- Results dir already exists at {self.config['save_dir']} ---")
    def initialize_network(self):
        if self.result_dict["ffm"] is None:
            # Creates the Fourier Feature Mapping
            self.B = BESTIEdata.fourier_feature_mapping.get_B(self.config["dataset"])
            if self.B is not None:
                self.input_size = 2 * int(self.config["dataset"]["fourier_feature_mapping"]["mapping_size"])
            else:
                self.input_size = len(self.config["dataset"]["input_vars"])

            print(f"--- The network has an input size of {self.input_size} ---")

            self.result_dict["ffm"] = {}
            self.result_dict["ffm"]["B"] = self.B

        print("--------------------- Initializing network ---------------------")
        if self.result_dict["params"] is None:
            # Initialize the network parameters
            if self.config["network"]["architecture"].lower() == "dense":
                init_params = self.net.init(self.rng,jnp.ones(self.input_size))["params"]
            elif self.config["network"]["architecture"].lower() == "transformer":
                init_params = self.net.init(self.rng,(jnp.ones((1,self.config["training"]["batch_size"],self.input_size))))["params"]
            self.rng, _ = random.split(self.rng)
            # Initialize the scale parameter to 0 which is needed for the number of bins
            # 0 corresponds to the number of bins which is set in the config
            init_params["scale"] = 0.
            init_params["Bscale"] = self.config["dataset"]["fourier_feature_mapping"]["scale"] #this is on a log scale

        else:
            init_params = self.result_dict["params"]

        self.result_dict["init_params"] = init_params

        if self.config["training"]["average_gradients"]:
            update_steps_per_epoch = 1
        else:
            update_steps_per_epoch = self.config["training"]["batches_per_epoch"]

        # Set up learning rate scheduling
        lr_fn = nets.lr_handler(self.config,update_steps_per_epoch)

        # Set up optimizer for network parameters
        tx = getattr(optax,self.config["training"]["optimizer"].lower())(learning_rate = lr_fn)

        class TrainState(train_state.TrainState):
            key: jax.Array
        self.rng, dropout_key = jax.random.split(self.rng)
        # Create state for network parameters
        self.state = TrainState.create(apply_fn=self.net.apply,
                                            params=init_params,
                                            key=dropout_key,
                                            tx=tx)

        def count_params(params):
            sizes = jax.tree_util.tree_map(lambda x: jnp.size(x), params)
            return sum(jax.tree_util.tree_leaves(sizes))

        num_params = count_params(self.state.params)
        print(f"🧠 Total number of parameters: {num_params}")
        
        # Define learning rate scheduling for bin training TODO: also implement this in the config
        def lr_fn_bins(*args,**kwargs):
            return lr_fn(*args,**kwargs)

        # Set up optimizer for bin parameters
        tx_bins = getattr(optax,self.config["training"]["optimizer"].lower())(learning_rate = lr_fn_bins) 

        # Setup state for bin parameters
        # Could this be combined with the state for the network parameters? The problem is the different learning rates
        self.state_bins = train_state.TrainState.create(apply_fn=self.net.apply,
                                            params=init_params,
                                            tx=tx_bins)
        
        # Create masks for the network and bin parameters
        self.train_mask = tree_map(lambda _: jnp.ones_like(_), init_params)
        self.train_mask["scale"] = False
        self.train_mask["Bscale"] = False

        self.bins_mask = tree_map(lambda _: jnp.zeros_like(_),init_params)
        assert isinstance(self.config["training"]["train_number_of_bins"],bool)
        self.bins_mask["scale"] = self.config["training"]["train_number_of_bins"]
        self.bins_mask["Bscale"] = True

    def load_data(self):
        self.df = pd.read_parquet(self.config["dataset"]["dataframe"])
        self.config["dataset"]["length"] = int(len(self.df))
        # Save one entry of the dataframe which will be needed to build the weight graph
        df_one = self.df[:1]
        df_one.to_parquet(os.path.join(self.config["save_dir"],"df_one.parquet"))

    def train_val_test_split(self, train_frac=0.5, val_frac=0.5, seed=0):
        self.rng, key = jax.random.split(self.rng)
        input_data, flux_vars, sample_weights = self.ds
        N = input_data.shape[0]

        # Shuffle indices
        indices = jax.random.permutation(key, N)

        # Compute split indices
        train_end = int(train_frac * N)
        val_end = int((train_frac + val_frac) * N)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        # Split input_data
        input_train = input_data[train_idx]
        input_val = input_data[val_idx]
        input_test = input_data[test_idx]

        # Split flux_vars
        flux_vars_train = {k: v[train_idx] for k, v in flux_vars.items()}
        flux_vars_val = {k: v[val_idx] for k, v in flux_vars.items()}
        flux_vars_test = {k: v[test_idx] for k, v in flux_vars.items()}

        # Split weights
        weights_train = sample_weights[train_idx]
        weights_val = sample_weights[val_idx]
        weights_test = sample_weights[test_idx]

        self.ds_train = (input_train, flux_vars_train, weights_train)
        self.ds_val   = (input_val,   flux_vars_val,   weights_val)
        self.ds_test  = (input_test,  flux_vars_test,  weights_test)
        
        self.weights_train = self.sample_weights_all[train_idx]
        self.weights_val = self.sample_weights_all[val_idx]
        self.weights_test = self.sample_weights_all[test_idx]

        print(f"Training Data: {len(self.ds_train[0])}")
        print(f"Validation Data: {len(self.ds_val[0])}")
        print(f"Test Data: {len(self.ds_test[0])}")

    def build_dataset(self):
        # Creating Pipeline Object
        
        weighter = partial(self.calc_weights,self.config["injected_params"].copy())
        self.ds, self.sample_weights_all, self.data_mask = BESTIEdata.make_jax_dataset(self.config,self.df,weighter=weighter)

    def load_results(self):
        if isinstance(self.result_dict, str):
            self.result_dict = jnp.load(self.result_dict,allow_pickle=True)

    def set_result_dict(self):
        if self.result_dict is None:
            self.result_dict = {}
            self.result_dict["history"] = []
            self.result_dict["losses"] = []
            self.result_dict["number_of_bins"] = []
            self.result_dict["params"] = None
            self.result_dict["learning_rate_epochs"] = []
            self.result_dict["ffm"] = None
            self.result_dict["val_loss"] = []

    def save_results(self):
        jnp.save(os.path.join(self.config["save_dir"],"result.pickle"),self.result_dict,allow_pickle=True)

    def build_train_step(self,ds,weights,training):
        noise_epsilon = float(self.config["training"].get("train_data_noise",0.))
        def l(params, batch, training, drop_out_key):
            data, aux, sample_weights = batch
            if self.config["dataset"]["fourier_feature_mapping"]["train_scale"]:
                Bdata = BESTIEdata.fourier_feature_mapping.input_mapping(data,self.B,params["Bscale"])
            else: 
                Bdata = BESTIEdata.fourier_feature_mapping.input_mapping(data,self.B,self.config["dataset"]["fourier_feature_mapping"]["scale"])
            loss, losses = self._optimization_pipeline(params,
                                            injected_params=Array(list(self.config["injected_params"].values())),
                                            data=Bdata,
                                            aux=aux,
                                            sample_weights=sample_weights,
                                            training=training,
                                            drop_out_key=drop_out_key
                                            )
            return loss, losses
        
        def _train_epoch(state, state_bins, rng):
            ### Do not call this function directly!
            # Loop over batches
            data, aux, sample_weights = ds

            if training:
                rng, noise_key= random.split(rng)
                
                data = data + noise_epsilon * random.normal(key=noise_key, shape=self.train_data_shape)
                #data = random.normal(key=noise_key,shape=())
            def step_fn(carry,batch_idx):
                state, state_bins, rng, accum_grads = carry
                rng, subkey, drop_out_key = random.split(rng,num=3)
                
                indices = random.choice(subkey, data.shape[0], shape=(self.config["training"]["batch_size"],),p=weights, replace=False)


                batch = data[indices], jax.tree_util.tree_map(lambda v: v[indices], aux), Array(1/sample_weights[indices] / jnp.sum(1/sample_weights[indices]) * data.shape[0])
                #Compute grads
                (loss, losses), grads = jax.value_and_grad(l, has_aux=True)(state.params, batch, training, drop_out_key)

                if self.config["training"]["average_gradients"]:
                    accum_grads = utilities.jax_utils.add_pytrees(accum_grads, grads)  # Accumulate
                else:
                    # Apply mask and update
                    grads_net = utilities.jax_utils.apply_mask(grads, self.train_mask)
                    grads_bins = utilities.jax_utils.apply_mask(grads, self.bins_mask)
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
            accum_grads = utilities.jax_utils.scale_pytrees(0., state.params)
            (state, state_bins, _, accum_grads), metrics = lax.scan(
                step_fn, (state, state_bins, init_key, accum_grads), jnp.arange(self.config["training"]["batches_per_epoch"])
            )

            if self.config["training"]["average_gradients"]:
                # Apply accumulated gradients
                grads_net = utilities.jax_utils.apply_mask(accum_grads, self.train_mask)
                grads_bins = utilities.jax_utils.apply_mask(accum_grads, self.bins_mask)
                state = state.apply_gradients(grads=grads_net)
                state_bins = state_bins.apply_gradients(grads=grads_bins)
                new_params = {**state.params, "scale": state_bins.params["scale"], "Bscale": state_bins.params["Bscale"]}
                state = state.replace(params=new_params)
            return state, state_bins, metrics, rng

        return jit(_train_epoch)

    def train_step(self,validate=False):
        try:
            print(f"--- Time to start training {time.time()-end_time:.2f} seconds ---")
        except:
            pass
        start_time = time.time()
        self.rng, _ = random.split(self.rng)
        self.state, self.state_bins, metrics, self.rng = self.train_epoch(self.state, self.state_bins, self.rng)
        print(f"--- Training step took {time.time()-start_time:.2f} seconds ---")
        start_time = time.time()
        self.log_metric(metrics,validate)
        print(f"--- Logging took {time.time()-start_time:.2f} seconds ---")
        end_time = time.time()
    
    def log_metric(self, metrics,validate=False):
        loss, losses = metrics
        loss = jnp.mean(loss)
        losses = jnp.mean(losses,axis=0)
        self.result_dict["history"].append(loss)
        self.result_dict["number_of_bins"].append(self.state.params["scale"])
        self.result_dict["losses"].append(losses)
        #self.result_dict["learning_rate_epochs"].append(self.state.optimizer.learning_rate)
        self.result_dict["params"] = self.state.params
        if validate==True:
            print("Validating...")
            self.rng, val_key = random.split(self.rng)
            _,_, metrics, self.rng = self.val_epoch(self.state, self.state_bins, val_key)
            val_loss, _ = metrics
            val_loss = jnp.mean(val_loss)
            self.result_dict["val_loss"].append(val_loss)
            print(f"Loss: {loss}, Val Loss: {val_loss}, Number of bins: {self.state.params['scale']}, Losses: {losses}")
        else:
            self.result_dict["val_loss"].append(jnp.nan)
            print(f"Loss: {loss}, Number of bins: {self.state.params['scale']}, Losses: {losses}")
        


    def inference(self,ds=None,bs=100_000,max_batches=-1):
        print(f"Processing {max_batches} batches")
        if ds is not None:
            data, aux, sample_weights = ds
        else:
            data, aux, sample_weights = self.ds_test
        lss_arr = []

        j = 1
        for i in tqdm(range(0,data.shape[0],bs)):
            
            batched_data = data[i:i+bs]
            if self.config["dataset"]["fourier_feature_mapping"]["train_scale"]:
                Bdata = BESTIEdata.fourier_feature_mapping.input_mapping(batched_data,self.B,self.state.params["Bscale"])
            else: 
                Bdata = BESTIEdata.fourier_feature_mapping.input_mapping(batched_data,self.B,self.config["dataset"]["fourier_feature_mapping"]["scale"])
            lss = self.calc_lss(self.state.params,Bdata)
            lss.block_until_ready()
            lss_arr.append(lss)
            if j == max_batches:
                print("Breaking")
                break
            j += 1
        lss_arr = jnp.concatenate(lss_arr,axis=0)
        return lss_arr

    def get_weights(self,ds=None,injected_params=None):
        if injected_params is None:
            injected_params = self.config["injected_params"].copy()
        if ds is not None:
            data, aux, sample_weights = ds
        else:
            data, aux, sample_weights = self.ds_test
        return self.calc_weights(injected_params,aux)
    
    def save_config(self):
        with open(os.path.join(self.config["save_dir"],"config.yaml"), 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)



    

