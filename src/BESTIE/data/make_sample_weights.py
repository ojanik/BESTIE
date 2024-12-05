import torch
from torch.utils.data import Dataset
import numpy as onp
import pandas as pd
import jax.numpy as jnp
import jax.scipy as jcp
Array = jnp.array

"""def calc_bin_idx(data):
    energy = Array(data[:,0])
    coszenith = jnp.cos(Array(data[:,1]))

    energy_bins = jnp.logspace(2,7,51)
    coszenith_bins = jnp.linspace(-1,0.0872,34)

    energy_digi = jnp.digitize(energy,energy_bins) - 1
    zenith_digi = jnp.digitize(coszenith,coszenith_bins) - 1
    bins_flattened = energy_digi * 33 + zenith_digi

    return bins_flattened"""

def calc_bin_idx(data):
    digi = []
    bins_arr = []

    number_of_sample_bins = 100
    
    for i in range(len(data.T)):
        var = Array(data[:,i])
        bins = jnp.linspace(jnp.min(var),jnp.max(var),number_of_sample_bins)
        bins_arr.append(bins)
        digi.append(jnp.digitize(var,bins,right=False)-1)

    for i, d in enumerate(digi):
        d[d < 0] = 0  # Clip indices below 0
        d[d >= len(bins[i]) - 1] = len(bins[i]) - 2  # Clip indices above the range

    bin_indices = jnp.ravel_multi_index(digi, [len(b) - 1 for b in bins_arr])
    return bin_indices
    

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
outfile = "/home/saturn/capn/capn105h/data/IceCube/simulation/torch_datasets/bin_sample_weights.pt"
df = pd.read_hdf(infile)


mask = create_mask(df,["energy_truncated_exists","reco_dir_exists"])
mask_energy_range = (Array(df["energy_truncated"] >= 10**2)) & (Array(df["energy_truncated"] <= 10**7))
mask_zenith_range = (jnp.cos(Array(df["zenith_MPEFit"])) >= -1) & (jnp.cos(Array(df["zenith_MPEFit"])) <= 0.0872)

mask = mask & mask_energy_range & mask_zenith_range

input_data = create_input_data(df,["energy_truncated","zenith_MPEFit","MCPrimaryEnergy",'powerlaw','mceq_conv_H4a_SIBYLL23c','mceq_conv_GST4_SIBYLL23c','mceq_pr_H4a_SIBYLL23c','mceq_pr_GST4_SIBYLL23c'],mask=mask)

data = onp.stack([df["energy_truncated"],df["zenith_MPEFit"]],axis=1)

bin_idx = calc_bin_idx(data[mask])

counts = onp.bincount(bin_idx)

sample_weights = 1/counts[bin_idx]

sample_weights = torch.tensor(sample_weights)

# Add galactic sampling
galactic_sample_weights = jcp.stats.norm.pdf(Array(df["lat_MPEFit"]),loc=0,scale=15)

sample_weights *= galactic_sample_weights

angular_uncert = jnp.log(df["L5_sigma_paraboloid"])
angular_uncert += jnp.min(angular_uncert)
angular_uncertainty_weights = jcp.stats.expon(angular_uncert)

sample_weights += angular_uncertainty_weights

print(len(sample_weights))
quit()

torch.save(sample_weights,outfile)


print(f"--- Saved sample weights dataset at {outfile}")