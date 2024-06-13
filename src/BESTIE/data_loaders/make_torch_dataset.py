from torch.utils.data import Dataset
import torch
import numpy as onp
import pandas as pd
import jax.numpy as jnp
Array = jnp.array

class SimpleDataset(Dataset):
    def __init__(self,input_data):
        self.input_data = input_data

        
        
    def __getitem__(self, idx):
        #data = {"reco_energy": self.data["energy_truncated"][idx],
        #       "reco_zenith": self.data["zenith_MPEFit"][idx]}
        data = torch.tensor([self.input_data[:,0][idx],self.input_data[:,1][idx]])
        
        aux = {"true_energy": self.input_data[:,2][idx],
                'mceq_pr_H4a_SIBYLL23c': self.input_data[:,6][idx],
                'mceq_pr_GST4_SIBYLL23c': self.input_data[:,7][idx],
                'powerlaw': self.input_data[:,3][idx],
                'mceq_conv_H4a_SIBYLL23c': self.input_data[:,4][idx],
                'mceq_conv_GST4_SIBYLL23c':self.input_data[:,5][idx]}
        
        return data, aux
    
    def __len__(self):
        return len(self.input_data)



def create_input_data(df,varis,mask=None):
    output = []
    for vari in varis:
        dtemp = onp.array(df[vari])
        if "energy_truncated" in vari.lower():
            dtemp = onp.log10(dtemp)
            dtemp = (dtemp-onp.mean(dtemp[mask]))/onp.std(dtemp[mask])
        if "zenith" in vari.lower():
            dtemp = onp.cos(dtemp)
            dtemp = (dtemp-onp.mean(dtemp[mask]))/onp.std(dtemp[mask])

        output.append(dtemp)
    
    output = onp.stack(output,axis=1)

    return output

def create_mask(df,exists):
    mask = onp.ones(len(df))
    for exist in exists:
        mask *= onp.array(df[exist] == 1)
    return onp.array(mask,dtype=bool)


infile = "/home/saturn/capn/capn105h/data/IceCube/simulation/NNMFit_dataframes/dataset_ds21002_ds21124_galactic.hdf"
outfile = "/home/saturn/capn/capn105h/data/IceCube/simulation/torch_datasets/full.pt"
df = pd.read_hdf(infile)


mask = create_mask(df,["energy_truncated_exists","reco_dir_exists"])
mask_energy_range = (Array(df["energy_truncated"] > 10**2)) & (Array(df["energy_truncated"] < 10**7))
mask_zenith_range = (jnp.cos(Array(df["zenith_MPEFit"])) > -1) & (jnp.cos(Array(df["zenith_MPEFit"])) < 0.0872)
mask = mask & mask_energy_range & mask_zenith_range
input_data = create_input_data(df,["energy_truncated","zenith_MPEFit","MCPrimaryEnergy",'powerlaw','mceq_conv_H4a_SIBYLL23c','mceq_conv_GST4_SIBYLL23c','mceq_pr_H4a_SIBYLL23c','mceq_pr_GST4_SIBYLL23c'],mask=mask)


ds = SimpleDataset(input_data[mask])

torch.save(ds,"/home/saturn/capn/capn105h/data/IceCube/simulation/torch_datasets/dataset.pt")