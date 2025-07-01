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
from jax import lax
import time
import pandas as pd
import yaml
import os

from ..pipeline import Optimization_Pipeline, Pipeline
from .. import utilities, nets
from .. import data as BESTIEdata
from ..data import Dataset


def has_nan(pytree):
    # Map each leaf to a boolean indicating presence of any NaNs in that leaf
    nan_trees = jax.tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x)), pytree)
    # Reduce the tree to a single boolean indicating if any leaf has NaNs
    return jax.tree_util.tree_reduce(lambda a, b: a | b, nan_trees)

class Train(Pipeline,Dataset):
    def __init__(self,config):
        Dataset.__init__(self, config)
        Pipeline.__init__(self, config)
        

        self.config = config
        self.result_dict = None

        self.rng = random.key(config["rng"])

        min_idx = int(0) 
        max_idx = int(self.config["train_split"] * self.len_input)
        print(f"Training split idx from {min_idx} to {max_idx}")
        print(f"Validation split idx from {max_idx+1} to {int(self.len_input-1)}")
        self.sample_val = self.get_sampler(max_idx+1,int(self.len_input-1))
        self.sample_train = self.get_sampler(min_idx,max_idx,)
        #self.sample_val = self.get_sampler(min_idx+1,max_idx-1)
        

        self._make_result_dir()
        self.set_result_dict()
        
        self.initialize_network(self.rng)
        rng = self.rerng(self.rng)

        self.train_epoch = self.build_train_step(training=True,sampler=self.sample_train,)
        self.val_epoch = self.build_train_step(training=False,sampler=self.sample_val)

    @staticmethod
    def rerng(rng):
        rng, _ = random.split(rng)
        return rng

    def initialize_network(self,rng):
        init_params = self.net.init(rng,jnp.ones(self.num_features))["params"]

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


    def build_train_step(self,training,sampler):
        
        def l(params, batch, rng):
            data, weights, grad_weights, sample_weights = batch
            loss, losses = self._optimization_pipeline(params,
                                                       data,
                                                       weights,
                                                       grad_weights,
                                                       sample_weights,
                                                       drop_out_key=rng
                                            )
            rng = self.rerng(rng)
            return loss, losses
        
        def _train_epoch(state, rng):
            ### Do not call this function directly!
            # Loop over batches

            def step_fn(carry,batch_idx):
                state, rng, accum_grads = carry
                rng, subkey, drop_out_key = random.split(rng,num=3)

                batch, rng = sampler(rng)

                rng, split_rng = random.split(rng)
                #Compute grads
                (loss, losses) , grads = jax.value_and_grad(l, has_aux=True)(state.params, batch, split_rng)

                if self.config["training"]["average_gradients"]:
                    accum_grads = utilities.jax_utils.add_pytrees(accum_grads, grads)  # Accumulate
                else:
                    # Apply mask and update
                    state = state.apply_gradients(grads=grads)


                carry = (state, rng, accum_grads)
                metrics = (loss, losses)
                return carry, metrics

            # Init
            rng, init_key = jax.random.split(rng)
            accum_grads = utilities.jax_utils.scale_pytrees(0., state.params)
            (state, _, accum_grads), metrics = lax.scan(
                step_fn, (state, init_key, accum_grads), jnp.arange(self.config["training"]["batches_per_epoch"])
            )

            if self.config["training"]["average_gradients"]:
                # Apply accumulated gradients
                state = state.apply_gradients(grads=accum_grads)
            return state, metrics, rng

        return jit(_train_epoch)

    def train_step(self,validate=False):
        try:
            print(f"--- Time to start training {time.time()-end_time:.2f} seconds ---")
        except:
            pass
        start_time = time.time()
        self.rng, _ = random.split(self.rng)
        self.state, metrics, self.rng = self.train_epoch(self.state, self.rng)
        print(f"--- Training step took {time.time()-start_time:.2f} seconds ---")
        start_time = time.time()
        self.log_metric(metrics,validate)
        print(f"--- Logging took {time.time()-start_time:.2f} seconds ---")
        end_time = time.time()
    
    def log_metric(self, metrics,validate=False):
        loss, losses = metrics
        loss = jnp.mean(loss)
        self.result_dict["history"].append(loss)
        #self.result_dict["learning_rate_epochs"].append(self.state.optimizer.learning_rate)
        self.result_dict["params"] = self.state.params
        if validate==True:
            print("Validating...")
            self.rng, val_key = random.split(self.rng)
            _, metrics, self.rng = self.val_epoch(self.state, self.rng)
            val_loss, _ = metrics
            val_loss = jnp.mean(val_loss)
            self.result_dict["val_loss"].append(val_loss)
            print(f"Loss: {loss}, Val Loss: {val_loss}")
        else:
            self.result_dict["val_loss"].append(jnp.nan)
            print(f"Loss: {loss}")
        