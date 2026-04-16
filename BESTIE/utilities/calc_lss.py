import BESTIE
import jax.numpy as jnp
Array = jnp.array
import os
from jax import jit, nn
from tqdm import tqdm
import numpy as onp
import pandas as pd


def calc_lss(model_path, df=None, disable_pbar=True):
    """Compute LSS values for a dataset using a trained BESTIE model.

    Parameters
    ----------
    model_path : str
        Path to the directory containing config.yaml and result.pickle.npy.
    df : pd.DataFrame, optional
        DataFrame to run inference on. If None, reads the path from the config.
    disable_pbar : bool
        Disable the tqdm progress bar during batched inference.

    Returns
    -------
    jnp.Array
        LSS values for the masked events.
    """
    path_to_config = os.path.join(model_path, "config.yaml")
    config = BESTIE.utilities.parse_yaml(path_to_config)

    model = BESTIE.nets.model_handler(config)
    net = model()

    if df is None:
        dataframe = config["dataset"]["dataframe"]
        _, ext = os.path.splitext(dataframe)
        print("--- reading dataframe ---")
        if ext[1:].lower() == "parquet":
            df = pd.read_parquet(dataframe)
        elif ext[1:].lower() in ("hdf", "hd5"):
            df = pd.read_hdf(dataframe)

    print("--- creating input data and masks ---")
    print("--- ignore warnings about dividing by zero, those events are filtered out in the end ---")
    input_data, mask_exists, mask_cut = BESTIE.data.prepare_data.create_input_data(df, config["dataset"])

    aux = {}
    for flux_var in config["dataset"]["flux_vars"]:
        dtemp = onp.array(df[flux_var])
        dtemp = dtemp[mask_exists & mask_cut]
        aux[flux_var] = dtemp

    # NNMFit needs true_energy
    if "MCPrimaryEnergy" in aux:
        print("Renamed key 'MCPrimaryEnergy' into 'true_energy'")
        aux["true_energy"] = aux.pop("MCPrimaryEnergy")

    results_dict = jnp.load(os.path.join(model_path, "result.pickle.npy"), allow_pickle=True)
    params = results_dict.item()["params"]
    B = results_dict.item()["ffm"]["B"]

    batch_size = 10000
    num_parts = int(jnp.ceil(len(input_data) / batch_size))
    apply_fn = jit(net.apply)

    print("--- Calculating lss ---")
    lss = None
    for i in tqdm(range(num_parts), disable=disable_pbar):
        batch = input_data[i*batch_size:(i+1)*batch_size, :len(config["dataset"]["input_vars"])]
        batch = BESTIE.data.fourier_feature_mapping.input_mapping(batch, B)
        out = apply_fn({"params": params}, batch)[:, 0]
        lss = out if lss is None else jnp.concatenate([lss, out])

    mask = mask_exists & mask_cut
    lss = lss[mask]

    kwargs = {"lss0": Array(df["lss0_standard_binning"])[mask]}

    init_params = results_dict.item()["init_params"]
    phi0 = None
    for i in tqdm(range(num_parts), disable=disable_pbar):
        batch = input_data[i*batch_size:(i+1)*batch_size, :len(config["dataset"]["input_vars"])]
        batch = BESTIE.data.fourier_feature_mapping.input_mapping(batch, B)
        out = apply_fn({"params": init_params}, batch)[:, 0]
        phi0 = out if phi0 is None else jnp.concatenate([phi0, out])

    kwargs["phi0"] = phi0[mask]

    transform_fun = BESTIE.transformations.transformation_handler(config["transformation"])
    lss = transform_fun(lss, **kwargs)
    bin_scale_up = config["hists"]["bins_up"] * nn.sigmoid(params["scale"]) * 2
    lss *= bin_scale_up

    return lss
