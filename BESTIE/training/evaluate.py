import os
import numpy as onp
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .train import Train
from ..utilities import parse_yaml


class Evaluate(Train):
    """Load a trained BESTIE model from disk and run inference / diagnostics.

    Parameters
    ----------
    result_dir : str
        Path to the directory produced by Train (contains config.yaml and result.pickle.npy).
    skip_inference : bool
        If True, skip running inference on load. Useful when you only need the
        model parameters or config, not the LSS values.
    overwrite_dataset : dict, optional
        Map of {dataset_name: new_dataframe_path} to swap out dataframes at
        evaluation time without editing the config on disk.
    """

    def __init__(self, result_dir, skip_inference=False, overwrite_dataset=None):
        config = parse_yaml(os.path.join(result_dir, "config.yaml"))

        if overwrite_dataset is not None:
            for hist_name, path in overwrite_dataset.items():
                config["datasets"][hist_name]["dataframe"] = path

        super().__init__(config, init_and_save=False)
        self.result_dir = result_dir
        self.load_results()

        if not skip_inference:
            self._run_inference()

    def load_results(self):
        """Load the saved result dict from disk."""
        self.result_dict = jnp.load(
            os.path.join(self.result_dir, "result.pickle.npy"), allow_pickle=True
        ).item()

    def _run_inference(self, bs=100_000, max_batches=None):
        """Run batched inference on all MC datasets and store results in self._lss_dict.

        Parameters
        ----------
        bs : int
            Batch size for inference.
        max_batches : int, optional
            Stop after this many batches. Useful for quick checks. None means all batches.
        """
        self._lss_dict = {
            dkey: self._batched_inference(
                self.datasets[dkey]["Dataset"].input_data,
                dkey,
                bs=bs,
                max_batches=max_batches,
            )
            for dkey in self.datasets
        }

    def get_lss_dict(self):
        """Return the inferred LSS arrays, keyed by dataset name."""
        return self._lss_dict

    def save_lss_to_dataframe(self):
        """Write the inferred LSS values back into each dataset's parquet file.

        Adds columns lss1, lss2, ... for each LSS dimension. Events outside the
        mask are left as NaN.
        """
        for dkey, lss in self._lss_dict.items():
            dfpath = self.config["datasets"][dkey]["dataframe"]
            df = pd.read_parquet(dfpath)
            D = self.datasets[dkey]["Dataset"]
            mask = pd.Series(onp.array(D.mask), index=df.index, dtype=bool)
            assert len(df) == len(mask), "Mask length mismatch — this should not happen."
            for i, col in enumerate(lss.T):
                col_name = f"lss{i + 1}"
                df[col_name] = onp.nan
                df.loc[mask, col_name] = onp.array(col)
            df.to_parquet(dfpath)
            print(f"Saved LSS columns to {dfpath}")

    def plot_2D_hists(self, weighted=False, save_dir=None):
        """Plot a 2D histogram of the two LSS dimensions for each dataset.

        Parameters
        ----------
        weighted : bool
            If True, weight events by their MC weights.
        save_dir : str, optional
            Directory to save plots. If None, plots are shown interactively.

        Raises
        ------
        AssertionError
            If the LSS array for any dataset does not have exactly 2 dimensions.
        """
        for dkey, lss in self._lss_dict.items():
            assert lss.shape[1] == 2, (
                f"plot_2D_hists requires 2D lss, got shape {lss.shape} for '{dkey}'"
            )
            hkey = self.hist_map[dkey]
            histd = self.config["hists"][hkey]["hists"]
            bins = jnp.linspace(histd["bins_low"], histd["bins_up"], histd["bins_number"] + 1)

            weights = None
            if weighted:
                weights = onp.array(self.datasets[dkey]["Dataset"].weights)

            h, _, _ = jnp.histogram2d(lss[:, 0], lss[:, 1],
                                      bins=[bins, bins], weights=weights)

            fig, ax = plt.subplots()
            im = ax.imshow(
                onp.array(h).T,
                origin="lower",
                aspect="auto",
                extent=[histd["bins_low"], histd["bins_up"],
                        histd["bins_low"], histd["bins_up"]],
            )
            fig.colorbar(im, ax=ax)
            ax.set_xlabel("lss dim 0")
            ax.set_ylabel("lss dim 1")
            ax.set_title(f"{dkey} ({'weighted' if weighted else 'unweighted'})")
            plt.tight_layout()

            if save_dir is not None:
                fname = os.path.join(save_dir, f"2D_hist_{dkey}.png")
                plt.savefig(fname, dpi=256)
                print(f"Saved {fname}")
            else:
                plt.show()
            plt.close()

    def get_sample_lss(self):
        """Draw one batch and return (batch, lss_dict)."""
        batch, self.rng = self.get_sample_dict(self.rng)
        lss_dict = self.calc_lss_dict(
            self.result_dict["params"], batch, self.hist_map,
            training=False, drop_out_key=self.rng,
        )
        return batch, lss_dict

    def get_test_hist(self):
        """Draw one sample batch and return the histogram dict."""
        _, lss_dict = self.get_sample_lss()
        hist_names = {k: self.hist_map[k] for k in lss_dict}
        return self.get_histograms(lss_dict, hist_names)
