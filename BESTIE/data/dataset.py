import jax.numpy as jnp
import numpy as onp
import pandas as pd
import jax
from jax import random

Array = jnp.array

from .prepare_data import create_input_data
from .sample_weights import sample_weight_handler

class Dataset():

    def __init__(self, config, dkey):
        self.config = config
        self.dkey = dkey
        hkey = config["datasets"][dkey]["hist"]
        self.type = config["datasets"][dkey]["type"]

        hconfig = config["hists"][hkey]
        self.livetime = hconfig["livetime"]

        self.hconfig = hconfig
        self.calc_sample_weights = sample_weight_handler(self.hconfig)

        dframe_path = config["datasets"][dkey]["dataframe"]
        df = pd.read_parquet(dframe_path)
        df = df.sample(frac=1, random_state=config["rng"]).reset_index(drop=True)
        self.input_data, self.mask = create_input_data(df, self.hconfig)
        _std_raw = (self._extract_standard_hist_data(df, hconfig["standard_hist"])
                    if "standard_hist" in hconfig else None)
        self.num_features = self.input_data.shape[1]
        
        # MC only vars
        # if data, fill with ones
        if self.type.lower() == "mc":
            self.weights = Array(df["weights"]) * self.livetime
            self.grad_weights = {}

            for key in df.keys():
                if "grad_" in key:
                    new_key = key.replace("grad_weights_", "")
                    self.grad_weights[new_key] = Array(df[key]) * self.livetime

        elif self.type.lower() == "data":
            self.weights = jnp.ones(len(df))
            self.grad_weights = {}

        else:
            raise ValueError(f"Type must either be data or mc, but is {self.type}")

        # === NaN Removal ===
        input_nan_mask = jnp.any(jnp.isnan(self.input_data), axis=1)
        weights_nan_mask = jnp.isnan(self.weights)
        


        grad_nan_mask = jnp.zeros_like(input_nan_mask)
        for v in self.grad_weights.values():
            grad_nan_mask |= jnp.isnan(v)

        total_nan_mask = input_nan_mask | weights_nan_mask | grad_nan_mask
        valid_mask = ~total_nan_mask

        combined_mask = valid_mask & self.mask
        self.input_data = self.input_data[combined_mask]
        self.weights = self.weights[combined_mask]
        self.standard_hist_data = Array(_std_raw[combined_mask]) if _std_raw is not None else None

        self.sample_weights = self.calc_sample_weights(self.input_data)
        self.mask = combined_mask
        for k in self.grad_weights:
            self.grad_weights[k] = self.grad_weights[k][valid_mask&self.mask]
        # Sort keys alphabetically as jax' tree operations will do it later anyway
        self.grad_weights = {k: self.grad_weights[k] for k in sorted(self.grad_weights)}

        print("number of nans removed: ",jnp.sum(total_nan_mask))
        print(f"number of events left: {len(self.input_data)},{self.mask.sum()}")
        self.len_input = len(self.input_data)

    @staticmethod
    def _extract_standard_hist_data(df, std_config):
        """Extract and scale variables for the standard histogram.

        Applies the same scaling as the main pipeline (log, cos, etc.) but
        skips the sphere normalisation so the values stay in physical units.
        The dataset mask is applied by the caller.
        """
        cols = []
        for var in std_config["vars"]:
            d = onp.array(df[var["var_name"]], dtype=onp.float64)
            if "scale" in var:
                try:
                    d = getattr(onp, var["scale"])(d)
                except AttributeError:
                    pass
            cols.append(d)
        return onp.stack(cols, axis=1)

    @staticmethod
    def rerng(rng):
        rng, _ = random.split(rng)
        return rng

    def get_sampler(self, min_idx, max_idx,smear=False):
        batch_size = self.config["datasets"][self.dkey]["batch_size"]
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
                p=sample_weights_draw, replace=True
            ) + min_idx

            x = input_data[indices]
            if smear:
                x = x + noise_epsilon * jax.random.normal(key=rng, shape=x.shape)
                rng = self.rerng(rng)
            sample_reweights = Array(1/sample_weights[indices] / jnp.sum(1/sample_weights[indices]) * len_input)

            return (
                x,
                weights[indices],
                {k: v[indices] for k, v in grad_weights.items()},
                sample_reweights,
            ), rng

        return sampler