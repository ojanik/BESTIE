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

def scan(pyff_path, save_path):
    import pyForwardFolding as pyFF
    PATH = pyff_path
    NFITS = 11
    plot_name = "scan"

    # Fixed directory for results
    results_dir = save_path
    output_path = os.path.join(results_dir, plot_name)

    # -----------------------------
    # Load dataset and preprocessing
    # -----------------------------
    dataset = pyFF.config.dataset_from_config(PATH)

    for k, d in dataset.items():
        mask = ~onp.isnan(d["galactic_baseline"])
        for kk, dd in d.items():
            if kk != "median_energy":
                dataset[k][kk] = dataset[k][kk][mask]

    # -----------------------------
    # Model parameters and priors
    # -----------------------------
    model_parameters = {
        "astro_norm": 1.44,
        "astro_index": 2.37,
        "atmo_norm": 1.0,
        "delta_gamma": 0.0,
        "prompt_norm": 1.0,
        "lambda_int": 0.0,
        "galactic_norm": 2.9,
        "barr_h": 0.0,
        "barr_w": 0.0,
        "barr_z": 0.0,
        "barr_y": 0.0,
    }

    prior_bounds_uni = {
        "astro_norm": (0., onp.inf),
        "prompt_norm": (0., onp.inf),
        "astro_index": (-10.0, 10.0),
        "delta_gamma": (-1, 1),
        "atmo_norm": (0., onp.inf),
        "lambda_int": (-1., 1.),
        "galactic_norm": (2.9, onp.inf),
    }

    prior_bounds_gauss = {
        "barr_h": (-0.8, 0.8),
        "barr_y": (-0.6, 0.6),
        "barr_w": (-0.6, 0.6),
        "barr_z": (-0.6, 0.6),
    }

    prior_seeds_uni = {
        "astro_norm": 1.44,
        "astro_index": 2.37,
        "atmo_norm": 1.0,
        "delta_gamma": 0.0,
        "prompt_norm": 1.0,
        "lambda_int": 0.0,
        "galactic_norm": 2.9,
    }

    prior_seeds_gauss = {
        "barr_h": 0.0,
        "barr_w": 0.0,
        "barr_z": 0.0,
        "barr_y": 0.0,
    }

    prior_params_gauss = {
        "barr_h": (0., .15),
        "barr_y": (0., .3),
        "barr_w": (0., 0.4),
        "barr_z": (0., 0.12),
    }

    priors = [
        pyFF.likelihood.UniformPrior(prior_seeds_uni, prior_bounds_uni),
        pyFF.likelihood.GaussianUnivariatePrior(
            prior_params_gauss, prior_seeds_gauss, prior_bounds_gauss
        ),
    ]

    # -----------------------------
    # Analysis setup
    # -----------------------------
    ana = pyFF.config.analysis_from_config(PATH)
    lik = pyFF.likelihood.PoissonLikelihood(ana, priors)
    likSAY = pyFF.likelihood.SAYLikelihood(ana, priors)

    hist, hist_ssq = ana.evaluate(dataset, model_parameters)

    # -----------------------------
    # Scan setup Poisson
    # -----------------------------
    Nfits = NFITS
    fixed_param = "galactic_norm"
    scan_points = onp.linspace(0, 5, Nfits)

    obs = hist
    hist_std = hist

    mini = pyFF.minimizer.ScipyMinimizer(lik)
    res = mini.minimize(obs, dataset, {})

    best_fit = res[1][fixed_param]

    obs = hist
    llhs = []
    pars = []
    results = []

    # -----------------------------
    # Perform likelihood scan
    # -----------------------------
    for scan_point in tqdm(scan_points):
        fixed_params = {fixed_param: scan_point}
        res, xmin, fun = mini.minimize(obs, dataset, fixed_params)
        results.append(res)
        llhs.append(fun)
        pars.append(xmin)

    # -----------------------------
    # Scan setup Poisson
    # -----------------------------
    Nfits = NFITS
    fixed_param = "galactic_norm"
    scan_points = onp.linspace(0, 5, Nfits)

    obs = hist
    hist_std = hist

    mini = pyFF.minimizer.ScipyMinimizer(likSAY)
    res = mini.minimize(obs, dataset, {})

    best_fit_SAY = res[1][fixed_param]

    obs = hist
    llhs_SAY = []
    pars_SAY = []
    results_SAY = []

    # -----------------------------
    # Perform likelihood scan
    # -----------------------------
    for scan_point in tqdm(scan_points):
        fixed_params = {fixed_param: scan_point}
        res, xmin, fun = mini.minimize(obs, dataset, fixed_params)
        results_SAY.append(res)
        llhs_SAY.append(fun)
        pars_SAY.append(xmin)

    # -----------------------------
    # Plot results
    # -----------------------------
    plt.plot(
        scan_points,
        2 * (onp.asarray(llhs) - min(llhs)),
        label="Poisson",
        color="r"
    )
    plt.plot(
        scan_points,
        2 * (onp.asarray(llhs_SAY) - min(llhs_SAY)),
        label="SAY",
        color="b"
    )
    plt.vlines(
        best_fit,
        0,
        4,
        color="r",
        linestyle="dashed",
        label="best fit standard"
    )
    plt.ylabel("-2 ΔLLH")
    plt.xlabel(fixed_param)
    plt.xlim(0, 5)
    plt.legend()
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_2d_hists(evaluator):
    lss_dict = evaluator.get_lss_dict()
    hist_dict = evaluator.get_test_hist()
    for k,v in evaluator.config["datasets"].items():
        dfpath = v["dataframe"]
        df = pd.read_parquet(dfpath)
        D = evaluator.datasets[k]["Dataset"]
        mask = D.mask

        fig, ax = plt.subplots(1,5,figsize=(20,4))
        weights = onp.array(df["weights"])[mask]
        gal_weights = onp.array(df["cringe_baseline"])[mask]

        hkey = evaluator.hist_map[k]
        binning = evaluator.config["hists"][hkey]["hists"]
        bins = onp.linspace(binning["bins_low"],binning["bins_up"],binning["bins_number"]+1)
        lss = lss_dict[k]
        Htot,_,_,im = ax[0].hist2d(lss[:,0],lss[:,1],bins=bins,weights=weights,norm=mcolors.LogNorm())
        ax[0].set_title("Total hist")
        fig.colorbar(im,ax=ax[0])
        Hgal,_,_,im = ax[1].hist2d(lss[:,0],lss[:,1],bins=bins,weights=gal_weights,norm=mcolors.LogNorm())
        ax[1].set_title("Galactic hist")
        fig.colorbar(im,ax=ax[1])
        im = ax[2].pcolormesh(Hgal.T/Htot.T,norm=mcolors.LogNorm())
        ax[2].set_title("Galactic Ratio")
        fig.colorbar(im,ax=ax[2])
        Hssq,_,_,im = ax[3].hist2d(lss[:,0],lss[:,1],bins=bins,weights=weights**2,norm=mcolors.LogNorm())
        ax[3].set_title("SSQ")
        fig.colorbar(im,ax=ax[3])
        im = ax[4].pcolormesh(onp.sqrt(Hssq.T)/Htot.T,norm=mcolors.LogNorm())
        ax[4].set_title("Rel uncert")
        try:
            fig.colorbar(im,ax=ax[4])
        except:
            print("Drawing colorbar for the Relative Uncertainty failed")

        plt.savefig(os.path.join(evaluator.config["save_dir"],f"gal_hist_{k}.png"))
        plt.close()
        # Compare hists

        fig, ax = plt.subplots(1,3,figsize=(12,4))
        mu = hist_dict[k]["mu"]
        mu = onp.reshape(mu,Htot.shape)
        im = ax[0].pcolormesh(mu.T,norm=mcolors.LogNorm())
        fig.colorbar(im,ax=ax[0])
        ax[0].set_title("Sampled")

        im = ax[1].pcolormesh(Htot.T,norm=mcolors.LogNorm())
        fig.colorbar(im,ax=ax[1])
        ax[1].set_title("Full")

        im = ax[2].pcolormesh(mu.T/Htot.T,norm=mcolors.LogNorm())
        fig.colorbar(im,ax=ax[2])
        ax[2].set_title("Ratio")

        plt.savefig(os.path.join(evaluator.config["save_dir"],f"compare_binning_{k}.png"))
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
        help="Path to a file or directory."
    )

    parser.add_argument(
        "--pyff",
        type=str,
        default=None,
        help="Path to a pyff config."
    )

    return parser.parse_args()

def main():
    args = parse_arguments()
    evaluate = Evaluate(args.path)

    print("Drawing loss curve")
    plot_loss_curve(evaluate)
    print("Drawing 2d hists")
    plot_2d_hists(evaluate)
    print("Saving lss to dataframes")
    save_to_df(evaluate)


    if args.pyff is not None:
        print("Scanning ...")
        scan(args.pyff,evaluate.config["save_dir"])


if __name__ == "__main__":
    main()
    print("Done")