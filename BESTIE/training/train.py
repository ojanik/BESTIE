print("Loaded training")
from datetime import datetime

from flax.training import train_state
import optax
from jax import random, jit
import jax.numpy as jnp
Array = jnp.array
import jax
from jax import lax
import time
import yaml
import os
from tqdm import tqdm

from ..pipeline import Pipeline
from .. import utilities, nets
from ..data import Dataset
from ..nets.train_state import MultiNetworkTrainState
from ..utilities import rearrange_matrix


def has_nan(pytree):
    # Map each leaf to a boolean indicating presence of any NaNs in that leaf
    nan_trees = jax.tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x)), pytree)
    # Reduce the tree to a single boolean indicating if any leaf has NaNs
    return jax.tree_util.tree_reduce(lambda a, b: a | b, nan_trees)


class Train(Pipeline):
    def __init__(self, config, name="unnamed", init_and_save=True, pretrained_params=None):
        Pipeline.__init__(self, config)

        self.config = config
        self.result_dict = None
        self._make_result_dir(name=name)
        self.rng = random.key(config["rng"])

        self.datasets, self.data_datasets, self.num_features = self._load_datasets(config)
        print(f"Num features: {self.num_features}")

        if init_and_save:
            self.initialize_network(self.rng, pretrained_params=pretrained_params)
            self.rng = self.rerng(self.rng)
            self.train_epoch = self.build_train_step(training=True)
            self.set_result_dict()
            self._compute_standard_hist_baseline()
            self.save_results()
            self.save_config()

    def _load_datasets(self, config):
        """Load all datasets from config, split into MC and data dicts."""
        mc_datasets = {}
        data_datasets = {}
        num_features = {}

        for dkey in config["datasets"]:
            print(f"Processing dataset {dkey}")
            D = Dataset(config, dkey)
            if D.type.lower() == "mc":
                max_idx = int(config["datasets"][dkey]["train_split"] * D.len_input)
                D.max_idx = max_idx
                mc_datasets[dkey] = {
                    "hist_name": config["datasets"][dkey]["hist"],
                    "sampler": D.get_sampler(0, max_idx, smear=True),
                    "Dataset": D,
                }
                num_features[config["datasets"][dkey]["hist"]] = D.num_features
            elif D.type.lower() == "data":
                data_datasets[dkey] = {
                    "hist_name": config["datasets"][dkey]["hist"],
                    "Dataset": D,
                }

        return mc_datasets, data_datasets, num_features

    @staticmethod
    def rerng(rng):
        rng, _ = random.split(rng)
        return rng

    @staticmethod
    def load_checkpoint_params(path, dtype=jnp.float32):
        """Load params from a checkpoint and cast to dtype.

        Accepts either the directory produced by Train or the full path to the
        result file (result.pickle.npy). Typical use: warm-starting a float32
        run from a float64 checkpoint.
            pretrained = Train.load_checkpoint_params("/path/to/run_dir/")
            trainer = Train(config, pretrained_params=pretrained)
        """
        if os.path.isdir(path):
            path = os.path.join(path, "result.pickle.npy")
        result = jnp.load(path, allow_pickle=True).item()
        return jax.tree_util.tree_map(lambda x: jnp.array(x, dtype=dtype), result["params"])

    def initialize_network(self, rng, pretrained_params=None):
        param_dict = {}
        apply_dict = {}
        for hkey in self.net_dict:
            if pretrained_params is not None:
                param_dict[hkey] = pretrained_params[hkey]
            else:
                init_params = self.net_dict[hkey]["net"].init(rng, jnp.ones(self.num_features[hkey]))
                param_dict[hkey] = init_params
            apply_dict[hkey] = self.net_dict[hkey]["net"].apply

        update_steps_per_epoch = (
            1 if self.config["training"]["average_gradients"]
            else self.config["training"]["batches_per_epoch"]
        )
        lr_fn = nets.lr_handler(self.config, update_steps_per_epoch)
        tx = getattr(optax, self.config["training"]["optimizer"].lower())(learning_rate=lr_fn)

        self.rng, key = jax.random.split(self.rng)
        self.state = MultiNetworkTrainState.create(
            apply_fns=apply_dict, params=param_dict, tx=tx, key=key
        )

        num_params = sum(jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(lambda x: jnp.size(x), self.state.params)
        ))
        print(f"🧠 Total number of parameters: {num_params}")

    def _make_result_dir(self, name="unnamed"):
        if "save_dir" not in self.config:
            date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_dir = os.path.join(self.config["output_dir"], f"{name}_{date_time_str}")
            self.config["save_dir"] = save_dir
            os.makedirs(save_dir, exist_ok=True)
            print(f"--- Results will be saved at {save_dir} ---")
        else:
            print(f"--- Results dir already exists at {self.config['save_dir']} ---")

    def set_result_dict(self):
        if self.result_dict is None:
            self.result_dict = {
                "history": [],
                "losses": [],
                "number_of_bins": [],
                "params": self.state.params,
                "learning_rate_epochs": [],
                "ffm": None,
                "val_loss": [],
                "mc_hists": [],
                "data_hists": [],
                "best_val_loss": jnp.inf,
                "best_params": None,
                "standard_hist_baseline": None,
            }

    def _get_opti_fn(self):
        """Return (opti_fn, extra_kwargs) matching the configured optimality."""
        lconfig = self.config["loss"]
        optimality = lconfig["optimality"].lower()

        if optimality in ["a", "a_optimality", "aoptimality"]:
            from ..losses.fisher_losses import A_optimality
            opti = A_optimality
        elif optimality in ["c", "c_optimality", "coptimality", "correlation"]:
            from ..losses.fisher_losses import C_optimality
            opti = C_optimality
        elif optimality in ["d", "d_optimality", "doptimality", "ellipsoid",
                            "uncertainty_ellipsoid", "ellipsoid_volume",
                            "uncertainty_ellipsoid_volume"]:
            from ..losses.fisher_losses import D_optimality
            opti = D_optimality
        elif optimality in ["m", "m_optimality", "moptimality", "ac"]:
            from ..losses.fisher_losses import M_optimality
            opti = M_optimality
        else:
            raise NotImplementedError(f"Optimality '{optimality}' not implemented")

        kwargs = {
            "weight_norm": lconfig.get("weight_norm", None),
        }
        # M_optimality also accepts alpha/beta
        if "alpha" in lconfig:
            kwargs["alpha"] = lconfig["alpha"]
        if "beta" in lconfig:
            kwargs["beta"] = lconfig["beta"]
        return opti, kwargs

    def _fim_to_loss(self, fisher_information, keys):
        """Apply Schur complement and call the configured opti function.

        Returns:
            sigmas_dict  – {param: sigma} from the *full* FIM diagonal (all params)
            loss_value   – scalar float from the configured optimality on the
                           Schur complement (marginalises nuisance parameters)
        """
        opti, opti_kwargs = self._get_opti_fn()
        signal_params = self.config["loss"].get("parameters_to_optimize", keys)
        fim_reg = self.config["training"].get("fim_regularization", 0.0)

        fish = fisher_information + fim_reg * jnp.eye(len(keys))
        cov = jnp.linalg.inv(fish)
        sigmas_dict = {keys[i]: float(jnp.sqrt(jnp.diag(cov)[i])) for i in range(len(keys))}

        signal_idx = [keys.index(p) for p in signal_params if p in keys]
        fish_rearranged = rearrange_matrix(fish, signal_idx)

        k = len(signal_idx)
        if k == len(keys):
            # All parameters are signal — no nuisance to marginalise
            S = fish_rearranged
        else:
            A = fish_rearranged[:k, :k]
            B = fish_rearranged[:k, k:]
            C = fish_rearranged[k:, k:]
            S = A - B @ jnp.linalg.inv(C) @ B.T

        loss_value = float(opti(S, **opti_kwargs))
        return sigmas_dict, loss_value

    def _compute_standard_hist_baseline(self):
        """Compute FIM once using the standard (fixed) histogram defined in the config.

        Called once at init. Stores per-dataset results in
        result_dict["standard_hist_baseline"] so validate() can report
        improvement ratios throughout training.
        """
        baselines = {}
        for dkey in self.datasets:
            hkey = self.hist_map[dkey]
            D = self.datasets[dkey]["Dataset"]
            if D.standard_hist_data is None:
                continue

            std_config = self.config["hists"][hkey]["standard_hist"]
            default_bins = std_config.get("bins_per_dim", 20)

            bins_nd = []
            for i, var in enumerate(std_config["vars"]):
                # Per-variable "bins" takes priority, then bins_per_dim list/scalar
                if "bins" in var:
                    n = var["bins"]
                elif isinstance(default_bins, list):
                    n = default_bins[i]
                else:
                    n = default_bins
                bins_nd.append(jnp.linspace(var["range"][0], var["range"][1], n + 1))

            split_correction = D.input_data.shape[0] / (D.input_data.shape[0] - D.max_idx)
            std_data = D.standard_hist_data[D.max_idx:]
            weights = D.weights[D.max_idx:] * split_correction
            grad_weights = {k: split_correction * v[D.max_idx:] for k, v in D.grad_weights.items()}

            mu, _ = jnp.histogramdd(std_data, bins=bins_nd, weights=jnp.array(weights))
            mu = mu.flatten()

            grad_hist = {}
            for k, gw in grad_weights.items():
                g, _ = jnp.histogramdd(std_data, bins=bins_nd, weights=jnp.array(gw))
                grad_hist[k] = g.flatten() / jnp.sqrt(mu + 1e-8)

            values = jnp.array(list(grad_hist.values()))
            keys = list(grad_hist.keys())
            fim = jnp.einsum('ib,jb->ij', values, values)
            sigmas_dict, loss_value = self._fim_to_loss(fim, keys)
            baselines[dkey] = {"sigmas": sigmas_dict, "loss": loss_value}
            print(f"Standard hist baseline ({dkey}): sigmas={sigmas_dict}, loss={loss_value:.6f}")

        self.result_dict["standard_hist_baseline"] = baselines

    def get_sample_dict(self, rng):
        batch = {}
        for dkey in self.datasets:
            b, rng = self.datasets[dkey]["sampler"](rng)
            data, weights, grad_weights, sample_weights = b
            batch[dkey] = {
                "data": data,
                "weights": weights,
                "grad_weights": grad_weights,
                "sample_weights": sample_weights,
            }
        return batch, rng

    def build_train_step(self, training):
        def _compute_loss(params, batch, rng):
            loss, losses = self._optimization_pipeline(params, batch, drop_out_key=rng)
            return loss, losses

        def _train_epoch(state, rng):
            """Run one full epoch (scanned over batches). Do not call directly — use train_step."""
            def step_fn(carry, _):
                state, rng, accum_grads = carry
                batch, rng = self.get_sample_dict(rng)
                rng, split_rng = random.split(rng)
                (loss, losses), grads = jax.value_and_grad(_compute_loss, has_aux=True)(
                    state.params, batch, split_rng
                )
                if self.config["training"]["average_gradients"]:
                    accum_grads = utilities.jax_utils.add_pytrees(accum_grads, grads)
                else:
                    state = state.apply_gradients(grads=grads)
                return (state, rng, accum_grads), (loss, losses)

            rng, init_key = jax.random.split(rng)
            accum_grads = utilities.jax_utils.scale_pytrees(0., state.params)
            (state, _, accum_grads), metrics = lax.scan(
                step_fn, (state, init_key, accum_grads),
                xs=None,
                length=self.config["training"]["batches_per_epoch"],
            )
            if self.config["training"]["average_gradients"]:
                state = state.apply_gradients(grads=accum_grads)
            return state, metrics, rng

        return jit(_train_epoch)

    def train_step(self, validate=False):
        if hasattr(self, "_last_step_end"):
            print(f"--- Time since last step: {time.time() - self._last_step_end:.2f}s ---")
        start_time = time.time()
        self.rng, _ = random.split(self.rng)
        self.state, metrics, self.rng = self.train_epoch(self.state, self.rng)
        print(f"--- Training step took {time.time() - start_time:.2f}s ---")
        start_time = time.time()
        self.log_metric(metrics, validate)
        print(f"--- Logging took {time.time() - start_time:.2f}s ---")
        self._last_step_end = time.time()

    def log_metric(self, metrics, validate=False):
        loss, losses = metrics
        loss = jnp.mean(loss)
        self.result_dict["history"].append(loss)
        self.result_dict["params"] = self.state.params
        if validate:
            print("Validating...")
            val_diag = self.validate()
            print("Val diag: ", val_diag)
        else:
            self.result_dict["val_loss"].append(jnp.nan)
            print(f"Loss: {loss}")

    def _batched_inference(self, data, dkey, start=0, bs=100_000, max_batches=None):
        """Run inference on data[start:] in batches, return concatenated LSS array."""
        lss_arr = []
        for i, offset in enumerate(tqdm(range(0, data.shape[0] - start, bs))):
            if max_batches is not None and i >= max_batches:
                break
            batched = data[offset + start : offset + start + bs]
            lss = self.calc_lss(
                self.result_dict["params"], batched, self.hist_map, dkey,
                drop_out_key=self.rng, training=False,
            )
            lss.block_until_ready()
            lss_arr.append(lss)
        return jnp.concatenate(lss_arr, axis=0)

    def validate(self):
        bs = 100_000
        best_val_key = self.config["training"].get("best_val_key", None)

        for dkey in self.datasets:
            hkey = self.hist_map[dkey]
            D = self.datasets[dkey]["Dataset"]

            lss_arr = self._batched_inference(D.input_data, dkey, start=D.max_idx, bs=bs)

            split_correction_factor = D.input_data.shape[0] / (D.input_data.shape[0] - D.max_idx)
            weights = D.weights[D.max_idx:] * split_correction_factor
            grad_weights = {k: split_correction_factor * v for k, v in D.grad_weights.items()}

            bins_lss = jnp.linspace(
                self.config["hists"][hkey]["hists"]["bins_low"],
                self.config["hists"][hkey]["hists"]["bins_up"],
                self.config["hists"][hkey]["hists"]["bins_number"] + 1,
            )
            n_lss = lss_arr.shape[1]
            bins_nd = [bins_lss] * n_lss
            mu, _ = jnp.histogramdd(lss_arr, bins=bins_nd, weights=jnp.array(weights))
            self.result_dict["mc_hists"].append(mu)
            mu = mu.flatten()

            grad_hist = {}
            for k, gw in grad_weights.items():
                gw = jnp.array(gw)[D.max_idx:]
                g, _ = jnp.histogramdd(lss_arr, bins=bins_nd, weights=gw)
                grad_hist[k] = g.flatten() / jnp.sqrt(mu + 1e-8)

            values = jnp.array(list(grad_hist.values()))
            keys = list(grad_hist.keys())
            fisher_information = jnp.einsum('ib,jb->ij', values, values)

            val_sigmas, val_loss_value = self._fim_to_loss(fisher_information, keys)

            print(f"Val sigma:   {val_sigmas}")
            print(f"Val loss: {val_loss_value:.6f}")

            baseline = (self.result_dict["standard_hist_baseline"] or {}).get(dkey)
            if baseline:
                improvement = {k: baseline["sigmas"][k] / val_sigmas[k]
                               for k in val_sigmas if k in baseline["sigmas"]}
                print(f"Std  loss: {baseline['loss']:.6f}")
                print(f"Improvement over standard hist:   {improvement}")
            self.result_dict["val_loss"].append(val_sigmas)

            tracked_key = best_val_key if best_val_key in val_sigmas else next(iter(val_sigmas))
            if val_sigmas[tracked_key] < self.result_dict["best_val_loss"]:
                self.result_dict["best_val_loss"] = val_sigmas[tracked_key]
                self.result_dict["best_params"] = self.state.params

        for dkey in self.data_datasets:
            hkey = self.hist_map[dkey]
            D = self.data_datasets[dkey]["Dataset"]

            lss_arr = self._batched_inference(D.input_data, dkey, bs=bs)

            bins_lss = jnp.linspace(
                self.config["hists"][hkey]["hists"]["bins_low"],
                self.config["hists"][hkey]["hists"]["bins_up"],
                self.config["hists"][hkey]["hists"]["bins_number"] + 1,
            )
            n_lss = lss_arr.shape[1]
            mu, _ = jnp.histogramdd(lss_arr, bins=[bins_lss] * n_lss,
                                    weights=jnp.array(D.weights))
            self.result_dict["data_hists"].append(mu)

        return 0

    def save_results(self):
        jnp.save(os.path.join(self.config["save_dir"], "result.pickle"),
                 self.result_dict, allow_pickle=True)

    def save_config(self):
        with open(os.path.join(self.config["save_dir"], "config.yaml"), "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
