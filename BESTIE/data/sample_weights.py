import jax.numpy as jnp
Array = jnp.array
from scipy.spatial import KDTree
from functools import partial

def sample_weight_handler(dconfig):
    """
    Returns a sample weighting function based on the configuration.

    Args:
        dconfig (dict): The dataset configuration dictionary expected to have the key
                        'sample_weights', which must contain a 'method' field.
                        Optional fields depend on the method:
                            - "hist", "histogram", or "binned": can include 'number_of_sample_bins'
                            - "knn": can include 'k'

    Returns:
        Callable: A function or a partially-applied function for computing sample weights.

    Raises:
        NotImplementedError: If an unsupported method is specified.
    """
    method = dconfig["sample_weights"]["method"].lower()

    if method in {"uniform"}:
        return uniform_sample_weights

    elif method in {"hist", "histogram", "binned"}:
        number_of_sample_bins = dconfig["sample_weights"].get("number_of_sample_bins", 20)
        return partial(hist_sample_weights, number_of_sample_bins=number_of_sample_bins)

    elif method in {"knn"}:
        k = dconfig["sample_weights"].get("k", 16)
        return partial(knn_sample_weights, k=k)

    else:
        raise NotImplementedError(f"'{method}' is not a valid sample weight method.")


def uniform_sample_weights(data):
    """
    Assigns uniform weights to all samples in the dataset.

    Args:
        data (jax.numpy.ndarray): An N x D array of data points.

    Returns:
        jax.numpy.ndarray: A 1D array of shape (N,) with uniform sample weights summing to 1.
    """
    sample_weights = jnp.ones(len(data))
    sample_weights /= jnp.sum(sample_weights)  # Normalize to sum to 1
    return sample_weights

def hist_sample_weights(data, number_of_sample_bins = 20):
    """
    Assigns weights to data points based on their bin occupancy in a multi-dimensional histogram.
    Points in densely populated bins get lower weights, and vice versa.

    Args:
        data (jax.numpy.ndarray): An N x D array of data points.

    Returns:
        jax.numpy.ndarray: A 1D array of shape (N,) with histogram-based sample weights.
    """
    digi = []          # List to hold digitized (binned) coordinates for each dimension
    bins_arr = []      # List to store bin edges for each dimension

    for i in range(data.shape[1]):  # Iterate over dimensions
        var = Array(data[:, i])  # Convert column to JAX array
        bins = jnp.linspace(jnp.min(var), jnp.max(var), number_of_sample_bins)  # Bin edges
        bins_arr.append(bins)
        digitized = jnp.digitize(var, bins, right=False) - 1  # Assign bins, start from 0
        digitized = jnp.clip(digitized, 0, number_of_sample_bins - 2)  # Keep within bounds
        digi.append(digitized)

    # Stack digitized results into a 2D array of shape (D, N)
    digi = Array(digi)

    # Compute flattened 1D bin indices for all points
    bin_indices = jnp.ravel_multi_index(digi, [number_of_sample_bins - 1] * data.shape[1])

    # Count how many samples fall into each bin
    counts = jnp.bincount(bin_indices)

    # Assign weight inversely proportional to bin count
    sample_weights = 1 / counts[bin_indices]
    sample_weights /= jnp.sum(sample_weights)  # Normalize to sum to 1
    return sample_weights

def knn_sample_weights(data, k=16):
    """
    Computes sample weights using k-nearest-neighbors (kNN) density estimation.
    Points in dense regions receive lower weights, and vice versa.

    Args:
        data (jax.numpy.ndarray): An N x D array of data points.
        k (int): Number of neighbors to consider for density estimation (default is 16).

    Returns:
        jax.numpy.ndarray: A 1D array of shape (N,) with kNN-based sample weights.
    """
    print("Calculating kNN weights...")

    data_np = jnp.array(data)  # Convert to NumPy array for KDTree
    tree = KDTree(data_np)     # Build KDTree for fast neighbor queries

    # Get distances to the k+1 nearest neighbors (includes the point itself)
    dists, _ = tree.query(data_np, k=k+1)

    # Sum distances to approximate local density (excluding self)
    knn_density = jnp.sum(dists, axis=1)

    # Use inverse density as sample weights (scaled by data dimensionality)
    sample_weights = knn_density ** len(data[0])
    sample_weights /= jnp.sum(sample_weights)  # Normalize to sum to 1
    return jnp.array(sample_weights)