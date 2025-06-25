import jax.numpy as jnp
import pandas as pd
import jax 

Array = jnp.array

from .prepare_data import create_input_data
from .sample_weights import sample_weight_handler
from .fourier_feature_mapping import input_mapping, get_B

class Dataset():

    def __init__(self,config):
        self.config = config
        self.calc_sample_weights = sample_weight_handler(self.config)

        df = pd.read_parquet(config["dataframe"])
        self.input_data, self.mask = create_input_data(df,self.config)
        self.sample_weights = self.calc_sample_weights(self.input_data)

        self.len_input = jnp.sum(self.mask)

        self.weights = Array(df["weights"])
        self.grad_weights = {}

        for key in df.keys():
            if "grad_" in key:
                new_key = key[:5] #remove "grad_" from key name
                self.grad_weights[new_key] = Array(df[key])

        self.B = get_B(config)
        self.logscale = config["fourier_feature_mapping"]["logscale"]

        
        

    def get_sampler(self,min_idx,max_idx):
        B = self.B
        logscale = self.logscale
        batch_size = self.config["training"]["batch_size"]
        sample_weights = Array(self.sample_weights[min_idx:max_idx])
        len_input = self.len_input
        assert max_idx < len_input
        input_data = Array(self.input_data)
        weights = Array(self.weights)
        grad_weights = {k: Array(v) for k, v in self.grad_weights.items()}

        @jax.jit
        def sampler(rng):
            rng, subkey = jax.random.split(rng)
            indices = jax.random.choice(subkey, max_idx-min_idx, shape=(batch_size,), p=sample_weights, replace=False) + min_idx

            x = input_data[indices]

            # use B if it exists, otherwise identity mapping
            if B is not None:
                x = input_mapping(x, B, logscale)

            return (
                x,
                weights[indices],
                {k: v[indices] for k, v in grad_weights.items()},
                sample_weights[indices],
            ), rng

        return sampler
        


        


    