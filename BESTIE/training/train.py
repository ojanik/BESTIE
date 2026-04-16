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


def has_nan(pytree):
    # Map each leaf to a boolean indicating presence of any NaNs in that leaf
    nan_trees = jax.tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x)), pytree)
    # Reduce the tree to a single boolean indicating if any leaf has NaNs
    return jax.tree_util.tree_reduce(lambda a, b: a | b, nan_trees)


class Train(Pipeline):
    def __init__(self, config, name="unnamed", init_and_save=True):
        Pipeline.__init__(self, config)

        self.config = config
        self.result_dict = None
        self._make_result_dir(name=name)
        self.rng = random.key(config["rng"])

        self.datasets, self.data_datasets, self.num_features = self._load_datasets(config)
        print(f"Num features: {self.num_features}")

        if init_and_save:
            self.initialize_network(self.rng)
            self.rng = self.rerng(self.rng)
            self.train_epoch = self.build_train_step(training=True)
            self.set_result_dict()
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

    def initialize_network(self, rng):
        param_dict = {}
        apply_dict = {}
        for hkey in self.net_dict:
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
            }

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
            lss1, lss2 = lss_arr[:, 0], lss_arr[:, 1]
            mu, _, _ = jnp.histogram2d(lss1, lss2, bins=[bins_lss, bins_lss],
                                       weights=jnp.array(weights))
            self.result_dict["mc_hists"].append(mu)
            mu = mu.flatten()

            grad_hist = {}
            for k, gw in grad_weights.items():
                gw = jnp.array(gw)[D.max_idx:]
                g, _, _ = jnp.histogram2d(lss1, lss2, bins=[bins_lss, bins_lss], weights=gw)
                grad_hist[k] = g.flatten() / jnp.sqrt(mu + 1e-8)

            values = jnp.array(list(grad_hist.values()))
            keys = list(grad_hist.keys())
            fisher_information = jnp.einsum('ib,jb->ij', values, values)
            fim_reg = self.config["training"].get("fim_regularization", 1e-3)
            fisher_reg = fisher_information + fim_reg * jnp.eye(len(keys))
            cov = jnp.linalg.solve(fisher_reg, jnp.eye(len(keys)))
            val_loss = {keys[i]: jnp.sqrt(jnp.diag(cov)[i]) for i in range(len(keys))}
            print(val_loss)
            self.result_dict["val_loss"].append(val_loss)

            tracked_key = best_val_key if best_val_key in val_loss else next(iter(val_loss))
            if val_loss[tracked_key] < self.result_dict["best_val_loss"]:
                self.result_dict["best_val_loss"] = val_loss[tracked_key]
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
            lss1, lss2 = lss_arr[:, 0], lss_arr[:, 1]
            mu, _, _ = jnp.histogram2d(lss1, lss2, bins=[bins_lss, bins_lss],
                                       weights=jnp.array(D.weights))
            self.result_dict["data_hists"].append(mu)

        return 0

    def save_results(self):
        jnp.save(os.path.join(self.config["save_dir"], "result.pickle"),
                 self.result_dict, allow_pickle=True)

    def save_config(self):
        with open(os.path.join(self.config["save_dir"], "config.yaml"), "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
