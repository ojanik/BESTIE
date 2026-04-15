#!/usr/bin/env python3
"""
Template script that takes a path and does something with it.
"""

import jax
jax.config.update("jax_enable_x64", True)
from BESTIE.training.evaluate import Evaluate
import jax.numpy as jnp
import numpy as onp
import numpy as onp
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tqdm import tqdm

import pandas as pd

import argparse
import os

def plot_loss_curve(evaluator):
    plt.scatter(onp.arange(len(evaluator.result_dict["history"])),evaluator.result_dict["history"])
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(os.path.join(evaluator.config["save_dir"],"loss_curve.png"))
    plt.close()

def save_to_df(evaluator):
    lss_dict = evaluator.get_lss_dict()
    for k,v in evaluator.config["datasets"].items():
        dfpath = v["dataframe"]
        df = pd.read_parquet(dfpath)
        D = evaluator.datasets[k]["Dataset"]
        mask = D.mask
        mask = pd.Series(mask, index=df.index, dtype=bool)
        assert len(df) == len(mask), "Mask and df have a shape mismatch. This should not happen!"
        for i,lss in enumerate(lss_dict[k].T):
            print(k,i,onp.min(lss),onp.max(lss))
            df[f"lss{i+1}"] = onp.nan
            df.loc[mask,f"lss{i+1}"] = lss

        df.to_parquet(dfpath)
        print(f"Saved dataframe with lss of {k} to {dfpath}")




def parse_arguments():
    """
    Set up command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Process a given path and do something with it."
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to a BESTIE result dir."
    )

    return parser.parse_args()

def main():
    args = parse_arguments()
    evaluate = Evaluate(args.path)
    print("Saving lss to dataframes")
    save_to_df(evaluate)
    print("Drawing loss curve")
    plot_loss_curve(evaluate)


if __name__ == "__main__":
    main()
    print("Done")