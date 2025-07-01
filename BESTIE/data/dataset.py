import jax.numpy as jnp
import pandas as pd
import jax

Array = jnp.array

from .prepare_data import create_input_data
from .sample_weights import sample_weight_handler
from .fourier_feature_mapping import input_mapping, get_B

class Dataset():

    def __init__(self, config):

        self.livetime = 1 * 287 * 24 * 3600

        self.config = config
        self.calc_sample_weights = sample_weight_handler(self.config)

        df = pd.read_parquet(config["dataframe"])
        df = df.sample(frac=1) # shuffle the dataframe
        self.input_data, self.mask = create_input_data(df, self.config)
        self.num_features = self.input_data.shape[1]
        self.sample_weights = self.calc_sample_weights(self.input_data)

        self.weights = Array(df["weights"]) * self.livetime
        self.grad_weights = {}

        for key in df.keys():
            if "grad_" in key:
                new_key = key.replace("grad_weights_", "")
                self.grad_weights[new_key] = Array(df[key]) * self.livetime

        # === NaN Removal ===
        input_nan_mask = jnp.any(jnp.isnan(self.input_data), axis=1)
        weights_nan_mask = jnp.isnan(self.weights)
        sample_weights_nan_mask = jnp.isnan(self.sample_weights)

        grad_nan_mask = jnp.zeros_like(input_nan_mask)
        for v in self.grad_weights.values():
            grad_nan_mask |= jnp.isnan(v)

        total_nan_mask = input_nan_mask | weights_nan_mask | sample_weights_nan_mask | grad_nan_mask
        valid_mask = ~total_nan_mask

        self.input_data = self.input_data[valid_mask]
        self.weights = self.weights[valid_mask]
        self.sample_weights = self.sample_weights[valid_mask]
        self.mask = self.mask[valid_mask]
        for k in self.grad_weights:
            self.grad_weights[k] = self.grad_weights[k][valid_mask]
        print("number of nans removed: ",jnp.sum(total_nan_mask))
        print(f"number of events left: {len(self.input_data)}")
        self.len_input = len(self.input_data)

        self.B = get_B(config)
        if self.B is not None:
            self.num_features = 2 * config["fourier_feature_mapping"]["mapping_size"]
        self.logscale = config["fourier_feature_mapping"]["logscale"]

    def get_sampler(self, min_idx, max_idx,smear=False):
        B = self.B
        logscale = self.logscale
        batch_size = self.config["training"]["batch_size"]
        sample_weights_draw = jnp.copy(Array(self.sample_weights[min_idx:max_idx]))
        sample_weights = Array(self.sample_weights)
        len_input = self.len_input
        assert max_idx < len_input
        input_data = Array(self.input_data)
        weights = Array(self.weights)
        grad_weights = {k: Array(v) for k, v in self.grad_weights.items()}
        noise_epsilon = float(self.config["training"].get("train_data_noise",0.))

        @jax.jit
        def sampler(rng):
            rng, subkey = jax.random.split(rng)
            indices = jax.random.choice(
                subkey, max_idx - min_idx, shape=(batch_size,),
                p=sample_weights_draw, replace=False
            ) + min_idx

            x = input_data[indices]
            if smear:
                x = x + noise_epsilon * jax.random.normal(key=rng, shape=x.shape)
                rng = self.rerng(rng)
            if B is not None:
                x = input_mapping(x, B, logscale)
            sample_reweights = Array(1/sample_weights[indices] / jnp.sum(1/sample_weights[indices]) * len_input)

            return (
                x,
                weights[indices],
                {k: v[indices] for k, v in grad_weights.items()},
                sample_reweights,
            ), rng

        return sampler