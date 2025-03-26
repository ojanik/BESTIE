import numpy as onp


import jax.numpy as jnp
Array = jnp.array

def create_input_data(df,config):
    mask_exists = onp.ones(len(df),dtype=bool)
    for exist in config["exists_vars"]:
        mask_exists &= onp.array(df[exist] == 1,dtype=bool)

    mask_cut = onp.ones(len(df),dtype=bool)
    
    for cut in config["cut_vars"]:
        mask_cut &= (Array(df[cut["var_name"]] >= cut["min"])) 
        mask_cut &= (Array(df[cut["var_name"]] <= cut["max"]))

    mask = mask_exists & mask_cut

    output = []
    for vari in config["input_vars"]:
        dtemp = onp.array(df[vari["var_name"]])

        if "oversample" in vari:
            print("oversample is not supported yet. Continuing without oversampling the data.")
            #dtemp = (dtemp+vari["oversample"])%(2*onp.pi)

        if "scale" in vari:
            try:
                print(f"Scaling {vari['var_name']} with {vari['scale']}")
                dtemp = getattr(onp, vari["scale"])(dtemp)
            except:
                print("Couldn't find given scale method. Continuing without scaling the data.")

        if "standardize" in vari:
            raise ValueError("standardize is not supported anymore. Use transform instead.")
            

        if vari["transform"] in ["standardize"]:
            dtemp = (dtemp-onp.mean(dtemp[mask]))/onp.std(dtemp[mask])

        elif vari["transform"] in ["sphere"]:
            dtemp -= jnp.min(dtemp[mask])
            dtemp /= (jnp.max(dtemp[mask]) + 1e-3) #small constant to not get 1 as input value

        output.append(dtemp)
    
    output = onp.stack(output,axis=1)


    return output, mask_exists, mask_cut

# def calc_bin_idx(data):
#     energy = Array(data[:,0])
#     coszenith = jnp.cos(Array(data[:,1]))

#     energy_bins = jnp.logspace(2,7,51)
#     coszenith_bins = jnp.linspace(-1,0.0872,34)

#     energy_digi = jnp.digitize(energy,energy_bins) - 1
#     zenith_digi = jnp.digitize(coszenith,coszenith_bins) - 1
#     bins_flattened = energy_digi * 33 + zenith_digi

#     return bins_flattened

def calc_bin_idx(data):
    digi = []
    bins_arr = []
    number_of_sample_bins = 20

    for i in range(data.shape[1]):  # Iterate over columns
        var = Array(data[:, i])  # Convert column to JAX array
        bins = jnp.linspace(jnp.min(var), jnp.max(var), number_of_sample_bins)
        bins_arr.append(bins)
        digitized = jnp.digitize(var, bins, right=False) - 1  # Bin indices start from 0
        digitized = jnp.clip(digitized, 0, number_of_sample_bins - 2)  # Clip indices within range
        digi.append(digitized)

    # Convert list of arrays into a JAX array
    digi = Array(digi)

    # Compute the flattened bin indices
    bin_indices = jnp.ravel_multi_index(digi, [number_of_sample_bins - 1] * data.shape[1])

    return bin_indices

import jax.numpy as jnp
import jax
from scipy.spatial import KDTree

def compute_knn_weights(data, k=16):
    """
    Compute kNN-based weights for a dataset to flatten a 1D projection.
    
    Args:
        data (jax.numpy.ndarray): N x D array of data points.
        k (int): Number of nearest neighbors to use for density estimation.
    
    Returns:
        jax.numpy.ndarray: Array of weights for each data point.
    """
    print("Calculating kNN weights...")

    data_np = onp.array(data)  # Convert to NumPy for KDTree
    tree = KDTree(data_np)  # Build KDTree

    # Find the distance to the k-th nearest neighbor
    dists, _ = tree.query(data_np, k=k+1)  # k+1 because first neighbor is itself
    knn_density = onp.sum(dists,axis=1)  # k-th nearest neighbor distance

    # Compute weights as inverse density
    weights = knn_density ** len(data[0]) # Add small epsilon to avoid division by zero
    return onp.array(weights)