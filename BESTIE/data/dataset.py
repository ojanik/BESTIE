import jax.numpy as jnp
import pandas as pd

from .prepare_data import create_input_data
from .sample_weights import sample_weight_handler

class Dataset():

    def __init__(self,dconfig, weighter=None):
        self.dconfig = dconfig
        self.calc_sample_weights = sample_weight_handler(self.dconfig)

        df = pd.read_parquet(dconfig["dataframe"])
        input_data, mask = create_input_data(df,self.dconfig)
        sample_weights = self.calc_sample_weights(input_data)

        


    