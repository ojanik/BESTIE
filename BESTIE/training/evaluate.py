import os
from tqdm import tqdm
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .train import Train
from ..utilities import parse_yaml
from ..data.fourier_feature_mapping import input_mapping


class Evaluate(Train):
    def __init__(self,result_dir):
        config = parse_yaml(os.path.join(result_dir,"config.yaml"))
        super().__init__(config)
        self.result_dir = result_dir
        self.load_results()

        self._run_inference()


    def load_results(self):
        self.result_dict = jnp.load(os.path.join(self.result_dir,"result.pickle.npy"),allow_pickle=True).item()

    def _run_inference(self,bs=100_000,max_batches=-1):
        print(f"Processing {max_batches} batches")
        lss_dict = {}
        for dkey in self.datasets.keys():
            D = self.datasets[dkey]["Dataset"]
            data = D.input_data
            lss_arr = []

            j = 1
            for i in tqdm(range(0,data.shape[0],bs)):
                
                batched_data = data[i:i+bs]
                batched_data = input_mapping(batched_data,D.B,D.logscale)

                lss = self.calc_lss(self.result_dict["params"],batched_data,self.hist_map,dkey,drop_out_key=self.rng,training=False)
                lss.block_until_ready()
                lss_arr.append(lss)
                if j == max_batches:
                    print("Breaking")
                    break
                j += 1
            lss_arr = jnp.concatenate(lss_arr,axis=0)
            lss_dict[dkey] = lss_arr
        self._lss_dict = lss_dict
    
    def get_lss_dict(self):
        return self._lss_dict

    def save_lss_to_dataframe(self):
        pass

    def plot_hists(self):
        pass


    