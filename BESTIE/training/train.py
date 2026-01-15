print("Loaded training")
from datetime import datetime

from flax.training import train_state
import optax
from jax import random, jit
import jax.numpy as jnp
Array = jnp.array
import jax 
from jax import lax
import time
import yaml
import os
from tqdm import tqdm

from ..pipeline import Pipeline
from .. import utilities, nets
from ..data import Dataset
from ..nets.train_state import MultiNetworkTrainState

def has_nan(pytree):
    # Map each leaf to a boolean indicating presence of any NaNs in that leaf
    nan_trees = jax.tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x)), pytree)
    # Reduce the tree to a single boolean indicating if any leaf has NaNs
    return jax.tree_util.tree_reduce(lambda a, b: a | b, nan_trees)

class Train(Pipeline):
    def __init__(self,config,name="unnamed"):
        #Dataset.__init__(self, config)
        Pipeline.__init__(self, config)
        

        self.config = config
        self.result_dict = None

        self.rng = random.key(config["rng"])

        self.datasets = {}
        self.num_features = {}
        for dkey in self.config["datasets"]:
            self.datasets[dkey] = {}
            self.datasets[dkey]["hist_name"] = self.config["datasets"][dkey]["hist"]
            D = Dataset(config,dkey)
            min_idx = int(0)
            max_idx = int(self.config["datasets"][dkey]["train_split"] * D.len_input)
            sampler = D.get_sampler(min_idx,max_idx,smear=True)
            self.datasets[dkey]["sampler"] = sampler
            self.datasets[dkey]["Dataset"] = D
            self.num_features[self.config["datasets"][dkey]["hist"]] = D.num_features
        
        

        self._make_result_dir(name=name)
        self.set_result_dict()
        
        self.initialize_network(self.rng)
        self.rng = self.rerng(self.rng)

        self.train_epoch = self.build_train_step(training=True)
        #self.val_epoch = self.build_train_step(training=False,sampler=self.sample_val)

        self.save_config()

    def get_data_dict_entry(self):
        pass


    @staticmethod
    def rerng(rng):
        rng, _ = random.split(rng)
        return rng

    def initialize_network(self,rng):
        apply_dict = {}
        param_dict = {}
        for hkey in self.net_dict.keys():
            init_params = self.net_dict[hkey]["net"].init(rng,jnp.ones(self.num_features[hkey]))
            param_dict[hkey] = init_params
            apply_dict[hkey] = self.net_dict[hkey]["net"].apply

        if self.config["training"]["average_gradients"]:
            update_steps_per_epoch = 1
        else:
            update_steps_per_epoch = self.config["training"]["batches_per_epoch"]

        # Set up learning rate scheduling
        lr_fn = nets.lr_handler(self.config,update_steps_per_epoch)

        # Set up optimizer for network parameters
        tx = getattr(optax,self.config["training"]["optimizer"].lower())(learning_rate = lr_fn)

        self.rng, key = jax.random.split(self.rng)
        self.state = MultiNetworkTrainState.create(apply_fns=apply_dict,
                                            params=param_dict,
                                            tx=tx,
                                            key=key)
        

        

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


    def get_sample_dict(self,rng):
        batch = {}
        for dkey in self.datasets:
            batch[dkey] = {}
            b, rng = self.datasets[dkey]["sampler"](rng)
            data, weights, grad_weights, sample_weights = b
            #batch[dkey]["hist_name"] = self.datasets[dkey]["hist_name"]
            batch[dkey]["data"] = data
            batch[dkey]["weights"] = weights
            batch[dkey]["grad_weights"] = grad_weights
            batch[dkey]["sample_weights"] = sample_weights
        return batch, rng

    def build_train_step(self,training):
        
        def l(params, batch, rng):
            loss, losses = self._optimization_pipeline(params,
                                                       batch,
                                                       drop_out_key=rng
                                            )
            rng = self.rerng(rng)
            return loss, losses
        
        def _train_epoch(state, rng):
            ### Do not call this function directly!
            # Loop over batches

            def step_fn(carry, _):
                state, rng, accum_grads = carry
                rng, subkey, drop_out_key = random.split(rng,num=3)


                batch , rng= self.get_sample_dict(rng)

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
                step_fn, (state, init_key, accum_grads),
                xs=jnp.arange(self.config["training"]["batches_per_epoch"]),
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
        self.result_dict["params"] = self.state.params
        #self.result_dict["learning_rate_epochs"].append(self.state.optimizer.learning_rate)
        
        if validate==True:
            print("Validating...")
            val_diag = self.validate()
            print("Val diag: ",val_diag)
        else:
            self.result_dict["val_loss"].append(jnp.nan)
            print(f"Loss: {loss}")

    def validate(self):
        # data = self.input_data
        # weights = self.weights
        # grad_weights = self.grad_weights
        # lss_arr = []
        bs = 100_000
        val_dict = {}
        for dkey in self.datasets.keys():
            hkey = self.hist_map[dkey]
            D = self.datasets[dkey]["Dataset"]
            data = D.input_data
            lss_arr = []

            j = 1
            for i in tqdm(range(0,data.shape[0],bs)):
                
                batched_data = data[i:i+bs]

                lss = self.calc_lss(self.result_dict["params"],batched_data,self.hist_map,dkey,drop_out_key=self.rng,training=False)
                
                lss.block_until_ready()
                lss_arr.append(lss)
                j += 1
            lss_arr = jnp.concatenate(lss_arr,axis=0)
            #lss_dict[dkey] = lss_arr

            weights = D.weights
            grad_weights = D.grad_weights


            
            lss1 = lss_arr[:,0]
            lss2 = lss_arr[:,1]
            bins_lss = jnp.linspace(self.config["hists"][hkey]["hists"]["bins_low"],self.config["hists"][hkey]["hists"]["bins_up"],self.config["hists"][hkey]["hists"]["bins_number"])
            mu, _, _ = jnp.histogram2d(lss1,lss2,bins=[bins_lss,bins_lss],weights=jnp.array(weights))
            mu = mu.flatten()
            grad_hist = {}
            for k in grad_weights:

                g, _, _ = jnp.histogram2d(lss1,lss2,bins=[bins_lss,bins_lss],weights=jnp.array(grad_weights[k]))
                g = g.flatten()
                g = g / jnp.sqrt(mu+1e-8)
                grad_hist[k] = g

            values = jnp.array([jnp.array(v) for v in grad_hist.values()])
            keys = [k for k in grad_hist.keys()]
            fisher_information = values[:, None, :] * values[None, :, :]
            fisher_information = jnp.sum(fisher_information,axis=-1)
            cov = jnp.linalg.inv(fisher_information)
            print({keys[i]:jnp.diag(cov)[i] for i in range(len(keys))})
            self.result_dict["val_loss"].append({keys[i]:jnp.diag(cov)[i] for i in range(len(keys))})
        return 0
    def save_results(self):
        jnp.save(os.path.join(self.config["save_dir"],"result.pickle"),self.result_dict,allow_pickle=True)
    
    def save_config(self):
        with open(os.path.join(self.config["save_dir"],"config.yaml"), 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)


        
