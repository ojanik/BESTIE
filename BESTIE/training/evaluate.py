import os
from tqdm import tqdm
import jax.numpy as jnp

from .train import Train
from ..utilities import parse_yaml
from ..data.fourier_feature_mapping import input_mapping


class Evaluate(Train):
    def __init__(self,result_dir):
        config = parse_yaml(os.path.join(result_dir,"config.yaml"))
        super().__init__(config)
        self.result_dir = result_dir
        self.load_results()


    def load_results(self):
        self.result_dict = jnp.load(os.path.join(self.result_dir,"result.pickle.npy"),allow_pickle=True).item()

    def inference(self,bs=100_000,max_batches=-1):
        print(f"Processing {max_batches} batches")
        data = self.input_data
        lss_arr = []

        j = 1
        for i in tqdm(range(0,data.shape[0],bs)):
            
            batched_data = data[i:i+bs]
            batched_data = input_mapping(batched_data,self.B,self.logscale)

            lss = self.calc_lss(self.result_dict["params"],batched_data,drop_out_key=self.rng,training=False)
            lss.block_until_ready()
            lss_arr.append(lss)
            if j == max_batches:
                print("Breaking")
                break
            j += 1
        lss_arr = jnp.concatenate(lss_arr,axis=0)
        return lss_arr


    