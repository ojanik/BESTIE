#!/usr/bin/env python
"""Visualise per-batch sampling for a BESTIE Dataset.

Builds a Dataset from a config, draws ``--n-batches`` batches through the
configured sampler, and writes a corner-plot PNG showing, for each pair of
input features:

  - full dataset distribution (gray, background)
  - drawn sample distribution (blue, foreground)
  - importance-reweighted sample (orange, dashed) — sanity check that the
    reweighting in ``Dataset.get_sampler`` restores the full distribution

Also prints per-feature 1D summaries and the effective sample size
``(Σw)^2 / Σw^2`` for the drawn batches.

Usage:
    python plot_sampling.py --config /path/to/config.yaml \\
        --dkey my_dataset \\
        --n-batches 4 \\
        --output sampling_diagnostic.png
"""
import argparse
import os
import sys

import numpy as onp
import matplotlib.pyplot as plt
import jax
from jax import random

# Allow running from anywhere as long as the BESTIE package is importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import BESTIE  # noqa: E402
from BESTIE.data import Dataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to config YAML.")
    p.add_argument("--dkey", required=True,
                   help="Dataset key in config['datasets'] to visualise.")
    p.add_argument("--n-batches", type=int, default=4,
                   help="Number of batches to draw (default: 4).")
    p.add_argument("--bins", type=int, default=60,
                   help="Histogram bins per axis (default: 60).")
    p.add_argument("--max-features", type=int, default=6,
                   help="Cap on input dimensions plotted (default: 6).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="sampling_diagnostic.png")
    p.add_argument("--no-reweighted", action="store_true",
                   help="Skip the importance-reweighted overlay.")
    return p.parse_args()


def draw_batches(D, dkey, config, n_batches, seed):
    """Run ``Dataset.get_sampler`` ``n_batches`` times and stack the results.

    Returns:
        x      : (B*n_batches, n_features) sampled input data
        sw_rew : (B*n_batches,) importance reweights (1/p_sample, normalised)
    """
    max_idx = int(config["datasets"][dkey]["train_split"] * D.len_input)
    sampler = D.get_sampler(0, max_idx, smear=False)

    rng = random.key(seed)
    xs, rws = [], []
    for _ in range(n_batches):
        (x, _w, _gw, sample_reweights), rng = sampler(rng)
        xs.append(onp.asarray(x))
        rws.append(onp.asarray(sample_reweights))
    return onp.concatenate(xs, axis=0), onp.concatenate(rws, axis=0)


def corner_plot(full, sampled, reweights, feature_names, bins, output,
                show_reweighted=True):
    """Lower-triangular corner plot. Diagonals: 1D hists. Off-diagonals: 2D
    contours (full dataset) + 2D hist (sampled events)."""
    d = full.shape[1]
    fig, axes = plt.subplots(d, d, figsize=(2.4 * d, 2.4 * d),
                             squeeze=False)

    # Per-feature bin edges from the full dataset, so all three histograms
    # share the same binning.
    edges = [onp.linspace(onp.min(full[:, i]), onp.max(full[:, i]), bins + 1)
             for i in range(d)]

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]

            if j > i:
                ax.set_visible(False)
                continue

            if i == j:
                # --- 1D marginal on the diagonal ---
                ax.hist(full[:, i], bins=edges[i], density=True,
                        color="0.7", alpha=0.6, label="full")
                ax.hist(sampled[:, i], bins=edges[i], density=True,
                        histtype="step", color="C0", lw=1.5, label="sampled")
                if show_reweighted:
                    ax.hist(sampled[:, i], bins=edges[i], density=True,
                            weights=reweights,
                            histtype="step", color="C1", lw=1.2,
                            linestyle="--", label="reweighted")
                ax.set_yticks([])
                if i == 0:
                    ax.legend(fontsize=7, frameon=False, loc="upper right")
            else:
                # --- 2D marginal off-diagonal ---
                # Full dataset as filled hist (background).
                ax.hist2d(full[:, j], full[:, i],
                          bins=[edges[j], edges[i]],
                          cmap="Greys", cmin=1)
                # Sampled events as contour lines.
                h_s, _, _ = onp.histogram2d(sampled[:, j], sampled[:, i],
                                            bins=[edges[j], edges[i]])
                # Smooth visually by plotting on cell centres.
                xc = 0.5 * (edges[j][:-1] + edges[j][1:])
                yc = 0.5 * (edges[i][:-1] + edges[i][1:])
                if h_s.sum() > 0:
                    # Few contour levels at sensible percentiles of the histogram mass.
                    levels = onp.quantile(h_s[h_s > 0], [0.5, 0.8, 0.95])
                    ax.contour(xc, yc, h_s.T, levels=levels,
                               colors="C0", linewidths=1.0)

            if i == d - 1:
                ax.set_xlabel(feature_names[j], fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(feature_names[i], fontsize=8)
            elif j == 0:
                pass
            else:
                ax.set_yticklabels([])

    fig.suptitle("Sampling diagnostic — full (grey) vs. drawn batches (blue)"
                 + ("  /  reweighted (orange dashed)" if show_reweighted else ""),
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main():
    args = parse_args()

    config = BESTIE.utilities.parse_yaml(args.config)
    if args.dkey not in config["datasets"]:
        raise SystemExit(
            f"Dataset '{args.dkey}' not in config['datasets'] "
            f"(have: {list(config['datasets'])})"
        )

    print(f"Loading dataset '{args.dkey}'...")
    D = Dataset(config, args.dkey)
    full = onp.asarray(D.input_data)
    n_feat = min(full.shape[1], args.max_features)
    if full.shape[1] > args.max_features:
        print(f"Note: dataset has {full.shape[1]} input features; "
              f"plotting first {n_feat} (override with --max-features).")
    full = full[:, :n_feat]

    print(f"Drawing {args.n_batches} batches via the configured sampler...")
    sampled, reweights = draw_batches(D, args.dkey, config,
                                      args.n_batches, args.seed)
    sampled = sampled[:, :n_feat]

    # ---- diagnostics ----
    n_eff = float((reweights.sum() ** 2) / (reweights ** 2).sum())
    print(f"\nDrew {len(sampled)} events across {args.n_batches} batches "
          f"(batch_size={config['datasets'][args.dkey]['batch_size']}).")
    print(f"N_eff (importance-sampling) = {n_eff:.0f}   "
          f"(uniform would give {len(sampled)})")
    print(f"  ratio N_eff / N = {n_eff / len(sampled):.3f}\n")

    feature_names = [f"x{i}" for i in range(n_feat)]
    # If the config carries human-readable input variable names, prefer those.
    hkey = config["datasets"][args.dkey]["hist"]
    input_vars = config["hists"][hkey].get("input_vars")
    if input_vars:
        names = [v.get("var_name", f"x{i}") for i, v in enumerate(input_vars)]
        feature_names = names[:n_feat]

    print("Building corner plot...")
    corner_plot(full, sampled, reweights, feature_names,
                bins=args.bins, output=args.output,
                show_reweighted=not args.no_reweighted)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
