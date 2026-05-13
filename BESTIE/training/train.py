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
        # Epoch counter used to evaluate the auxiliary score-head weight
        # schedule (``Pipeline.current_score_weight``). Incremented after each
        # call to ``train_step``. Resuming from a checkpoint that should pick
        # up mid-schedule needs to set this explicitly before the first step.
        self.epoch = 0
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
        optimizer_name = self.config["training"]["optimizer"].lower()
        optimizer_kwargs = self.config["training"].get("optimizer_kwargs", {})
        tx = getattr(optax, optimizer_name)(learning_rate=lr_fn, **optimizer_kwargs)

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
                "val_loss_scalar": [],
                "train_val_loss_scalar": [],
                # Combined-across-datasets validation metrics. Mirror the
                # per-dataset entries above but aggregate via FIM addition,
                # which is mathematically what the training loss does at the
                # bin-concatenation step. These let you track a single
                # convergence curve regardless of dataset count.
                "combined_val_loss": [],
                "combined_val_loss_scalar": [],
                "combined_train_val_loss_scalar": [],
                "mc_hists": [],
                "data_hists": [],
                "best_val_loss": jnp.inf,
                "best_params": None,
                "standard_hist_baseline": None,
                "unbinned_baseline": None,
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
        """Compute reference FIMs once at init using the full dataset.

        Three baselines are stored per dataset in result_dict:
          - "standard_hist_baseline": FIM from the fixed histogram defined in config.
          - "unbinned_baseline":      FIM from treating every event as its own bin
                                      (theoretical maximum for any histogram).
        """
        std_baselines = {}
        unbinned_baselines = {}

        for dkey in self.datasets:
            hkey = self.hist_map[dkey]
            D = self.datasets[dkey]["Dataset"]

            keys = sorted(D.grad_weights.keys())
            grad_matrix = jnp.stack([D.grad_weights[k] for k in keys], axis=1)  # (N, P)

            # --- Unbinned FIM (theoretical maximum) ---
            fim_unbinned = (grad_matrix / D.weights[:, None]).T @ grad_matrix
            sigmas_u, loss_u = self._fim_to_loss(fim_unbinned, keys)
            # `fim` and `keys` are kept so validate() can sum FIMs across
            # datasets to produce a "combined" baseline that is directly
            # comparable to the combined val/train losses.
            unbinned_baselines[dkey] = {
                "sigmas": sigmas_u, "loss": loss_u,
                "fim": fim_unbinned, "keys": keys,
            }
            print(f"Unbinned baseline      ({dkey}): sigmas={sigmas_u}, loss={loss_u:.6f}")

            # --- Standard histogram FIM ---
            if D.standard_hist_data is None:
                continue

            std_config = self.config["hists"][hkey]["standard_hist"]
            default_bins = std_config.get("bins_per_dim", 20)

            bins_nd = []
            for i, var in enumerate(std_config["vars"]):
                if "bins" in var:
                    n = var["bins"]
                elif isinstance(default_bins, list):
                    n = default_bins[i]
                else:
                    n = default_bins
                bins_nd.append(jnp.linspace(var["range"][0], var["range"][1], n + 1))

            mu, _ = jnp.histogramdd(D.standard_hist_data, bins=bins_nd,
                                    weights=jnp.array(D.weights))
            mu = mu.flatten()

            grad_hist = {}
            for k in keys:
                g, _ = jnp.histogramdd(D.standard_hist_data, bins=bins_nd,
                                       weights=jnp.array(D.grad_weights[k]))
                grad_hist[k] = g.flatten() / jnp.sqrt(mu + 1e-8)

            fim_std = jnp.einsum('ib,jb->ij',
                                 jnp.array(list(grad_hist.values())),
                                 jnp.array(list(grad_hist.values())))
            sigmas_s, loss_s = self._fim_to_loss(fim_std, keys)
            std_baselines[dkey] = {
                "sigmas": sigmas_s, "loss": loss_s,
                "fim": fim_std, "keys": keys,
            }
            print(f"Standard hist baseline ({dkey}): sigmas={sigmas_s}, loss={loss_s:.6f}")

        self.result_dict["standard_hist_baseline"] = std_baselines
        self.result_dict["unbinned_baseline"] = unbinned_baselines

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
        def _compute_loss(params, batch, rng, score_weight):
            loss, losses = self._optimization_pipeline(
                params, batch, score_weight, drop_out_key=rng,
            )
            return loss, losses

        def _train_epoch(state, rng, score_weight):
            """Run one full epoch (scanned over batches). Do not call directly — use train_step.

            ``score_weight`` is held constant across the batches of one epoch
            (the schedule advances per-epoch, not per-batch). It's threaded as
            a runtime JAX scalar so JIT does not retrace when its value
            changes between epochs.
            """
            def step_fn(carry, _):
                state, rng, accum_grads = carry
                batch, rng = self.get_sample_dict(rng)
                rng, split_rng = random.split(rng)
                # grad is taken wrt argument 0 (params) only, so passing
                # score_weight as an additional non-differentiated arg is safe.
                (loss, losses), grads = jax.value_and_grad(_compute_loss, has_aux=True)(
                    state.params, batch, split_rng, score_weight,
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
        # Evaluate the score-head decay schedule for the current epoch. With
        # the schedule disabled (or the score head off) this is a constant.
        score_weight = jnp.asarray(self.current_score_weight(self.epoch))
        if self.score_head_enabled:
            print(f"--- Score-head weight (epoch {self.epoch}): {float(score_weight):.6g} ---")
        self.state, metrics, self.rng = self.train_epoch(
            self.state, self.rng, score_weight,
        )
        self.epoch += 1
        print(f"--- Training step took {time.time() - start_time:.2f}s ---")
        start_time = time.time()
        self.log_metric(metrics, validate)
        print(f"--- Logging took {time.time() - start_time:.2f}s ---")
        self._last_step_end = time.time()

    def log_metric(self, metrics, validate=False):
        loss, losses = metrics
        loss = jnp.mean(loss)
        # Per-component loss breakdown averaged over the epoch's batches.
        # Shape: (n_components,). With the default Fisher-only setup this is
        # length 1 (or 2 if bin_loss is also enabled). With loss.score_head
        # enabled, the last entry is the auxiliary score-regression loss
        # (BEFORE the score_weight multiplier, so the raw aux value is
        # logged independently of how it's weighted into the total).
        losses_mean = jnp.mean(losses, axis=0)
        self.result_dict["history"].append(loss)
        self.result_dict["losses"].append(losses_mean)
        self.result_dict["params"] = self.state.params
        if validate:
            print("Validating...")
            val_diag = self.validate()
            print("Val diag: ", val_diag)
        else:
            self.result_dict["val_loss"].append(jnp.nan)
            self.result_dict["val_loss_scalar"].append(float("nan"))
            self.result_dict["train_val_loss_scalar"].append(float("nan"))
            print(f"Loss: {loss}  components={losses_mean}")

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

    def _compute_hard_fim(self, lss_arr, weights, grad_weights, hkey):
        """Compute FIM from a hard histogramdd on the given LSS array."""
        bins_lss = jnp.linspace(
            self.config["hists"][hkey]["hists"]["bins_low"],
            self.config["hists"][hkey]["hists"]["bins_up"],
            self.config["hists"][hkey]["hists"]["bins_number"] + 1,
        )
        bins_nd = [bins_lss] * lss_arr.shape[1]

        mu, _ = jnp.histogramdd(lss_arr, bins=bins_nd, weights=jnp.array(weights))
        mu = mu.flatten()

        grad_hist = {}
        for k, gw in grad_weights.items():
            g, _ = jnp.histogramdd(lss_arr, bins=bins_nd, weights=jnp.array(gw))
            grad_hist[k] = g.flatten() / jnp.sqrt(mu + 1e-8)

        keys = list(grad_hist.keys())
        values = jnp.array(list(grad_hist.values()))
        fim = jnp.einsum('ib,jb->ij', values, values)
        return fim, keys, mu

    @staticmethod
    def _format_sigma_block(label, val_sigmas, train_sigmas,
                             val_loss_value, train_loss_value,
                             std_baseline, unbinned):
        """Render a single (header line + per-parameter table) block.

        Used for both per-dataset and the aggregated "combined" view. Columns
        adapt to which baselines are available. Conventions:
          - val/std  < 1.0  → network beats standard binning (good)
          - val/unb  >= 1.0 → unbinned is the theoretical lower bound
        """
        has_std = std_baseline is not None
        has_unb = unbinned is not None

        # Scalar header line (the four numbers you most often want).
        head = [f"val={val_loss_value:.6f}", f"train={train_loss_value:.6f}"]
        if has_std:
            head.append(f"std={std_baseline['loss']:.6f}")
        if has_unb:
            head.append(f"unbinned={unbinned['loss']:.6f}")
        lines = [f"[{label}]   " + "  ".join(head)]

        # Column layout
        cols = ["param", "val σ", "train σ"]
        if has_std:
            cols += ["std σ", "val/std"]
        if has_unb:
            cols += ["unb σ", "val/unb"]

        name_w = max(12, max((len(k) for k in val_sigmas), default=12))
        num_w = 11

        header = f"  {cols[0]:<{name_w}}" + "".join(f"  {c:>{num_w}}" for c in cols[1:])
        lines.append(header)
        lines.append("  " + "-" * (name_w + (num_w + 2) * (len(cols) - 1)))

        for k in sorted(val_sigmas.keys()):
            row = [f"  {k:<{name_w}}",
                   f"  {val_sigmas[k]:>{num_w}.4g}",
                   f"  {train_sigmas[k]:>{num_w}.4g}"]
            if has_std:
                if k in std_baseline['sigmas']:
                    s = std_baseline['sigmas'][k]
                    row += [f"  {s:>{num_w}.4g}",
                            f"  {val_sigmas[k] / s:>{num_w - 1}.3f}x"]
                else:
                    row += [f"  {'n/a':>{num_w}}", f"  {'n/a':>{num_w}}"]
            if has_unb:
                if k in unbinned['sigmas']:
                    u = unbinned['sigmas'][k]
                    row += [f"  {u:>{num_w}.4g}",
                            f"  {val_sigmas[k] / u:>{num_w - 1}.3f}x"]
                else:
                    row += [f"  {'n/a':>{num_w}}", f"  {'n/a':>{num_w}}"]
            lines.append("".join(row))

        return "\n".join(lines)

    @staticmethod
    def _sum_fims_aligned(fim_list_with_keys):
        """Sum per-dataset FIMs over the union of their parameter keys.

        Each input FIM is zero-padded into the combined key space before
        summation, so datasets with different (or partially overlapping)
        parameter sets compose correctly. This mirrors what
        `pipeline._group_and_concat_hists` does at training time when it
        zero-pads missing keys before concatenating bins.

        Returns (fim_total, all_keys), or (None, []) if the input is empty.
        """
        if not fim_list_with_keys:
            return None, []
        all_keys = sorted(set().union(*(set(k) for _, k in fim_list_with_keys)))
        n = len(all_keys)
        fim_total = jnp.zeros((n, n))
        for fim, keys in fim_list_with_keys:
            idx = jnp.array([all_keys.index(k) for k in keys])
            fim_total = fim_total.at[jnp.ix_(idx, idx)].add(fim)
        return fim_total, all_keys

    def _build_combined_baseline(self, source_dict):
        """Sum per-dataset baseline FIMs into a single combined baseline.

        Aligns parameter sets via the union of keys across datasets (same
        zero-padding logic as the training-time bin concatenation). Returns
        {"sigmas": ..., "loss": ...} or None if no dataset stored a FIM
        (true for older result_dicts that pre-date this change).
        """
        if not source_dict:
            return None
        fims = []
        for dkey in self.datasets:
            entry = source_dict.get(dkey)
            if entry is None or "fim" not in entry:
                continue
            fims.append((entry["fim"], entry.get("keys")))
        if not fims:
            return None
        fim_total, all_keys = self._sum_fims_aligned(fims)
        sig, loss = self._fim_to_loss(fim_total, all_keys)
        return {"sigmas": sig, "loss": loss}

    def validate(self):
        bs = 100_000

        # Accumulators for the combined section. We sum FIMs (mathematically
        # equivalent to concatenating bins, which is how training combines
        # them). Datasets with mismatched parameter sets are reconciled by
        # taking the union of keys and zero-padding the missing entries —
        # the same logic `pipeline._group_and_concat_hists` uses at train
        # time, so the combined view is well-defined whenever training is.
        per_dataset_val_fims = []   # list of (fim, keys)
        per_dataset_train_fims = []
        blocks = []

        for dkey in self.datasets:
            hkey = self.hist_map[dkey]
            D = self.datasets[dkey]["Dataset"]
            N = D.input_data.shape[0]

            # --- Validation split ---
            lss_val = self._batched_inference(D.input_data, dkey, start=D.max_idx, bs=bs)
            val_correction = N / (N - D.max_idx)
            val_weights = D.weights[D.max_idx:] * val_correction
            val_grad_weights = {k: v[D.max_idx:] * val_correction
                                for k, v in D.grad_weights.items()}

            fim_val, keys, mu_val = self._compute_hard_fim(
                lss_val, val_weights, val_grad_weights, hkey)
            self.result_dict["mc_hists"].append(mu_val.reshape(
                [self.config["hists"][hkey]["hists"]["bins_number"]] * lss_val.shape[1]))
            val_sigmas, val_loss_value = self._fim_to_loss(fim_val, keys)

            # --- Training split ---
            lss_train = self._batched_inference(D.input_data, dkey, start=0,
                                                bs=bs, max_batches=None)
            lss_train = lss_train[:D.max_idx]
            train_correction = N / D.max_idx
            train_weights = D.weights[:D.max_idx] * train_correction
            train_grad_weights = {k: v[:D.max_idx] * train_correction
                                  for k, v in D.grad_weights.items()}

            fim_train, _, _ = self._compute_hard_fim(
                lss_train, train_weights, train_grad_weights, hkey)
            train_sigmas, train_loss_value = self._fim_to_loss(fim_train, keys)

            std_baseline = (self.result_dict["standard_hist_baseline"] or {}).get(dkey)
            unbinned = (self.result_dict["unbinned_baseline"] or {}).get(dkey)

            # Build the human-readable block now; print everything together
            # at the end so the per-dataset tables aren't fragmented across
            # the inference progress bars.
            blocks.append(self._format_sigma_block(
                dkey, val_sigmas, train_sigmas,
                val_loss_value, train_loss_value,
                std_baseline, unbinned,
            ))

            per_dataset_val_fims.append((fim_val, keys))
            per_dataset_train_fims.append((fim_train, keys))

            self.result_dict["val_loss"].append(val_sigmas)
            self.result_dict["val_loss_scalar"].append(float(val_loss_value))
            self.result_dict["train_val_loss_scalar"].append(float(train_loss_value))

        # --- Combined block (FIMs summed over the union of param keys) ---
        combined_val_sigmas = None
        combined_val_loss = float("nan")
        combined_train_loss = float("nan")
        combined_block = None

        if per_dataset_val_fims:
            fim_val_sum, combined_keys = self._sum_fims_aligned(per_dataset_val_fims)
            fim_train_sum, _ = self._sum_fims_aligned(per_dataset_train_fims)
            combined_val_sigmas, combined_val_loss = self._fim_to_loss(
                fim_val_sum, combined_keys)
            combined_train_sigmas, combined_train_loss = self._fim_to_loss(
                fim_train_sum, combined_keys)

            # Combined baselines, also union-aligned. Absent from runs that
            # pre-date storing the FIM in the baseline dict — handled inside
            # _build_combined_baseline by returning None.
            combined_std = self._build_combined_baseline(
                self.result_dict.get("standard_hist_baseline"))
            combined_unb = self._build_combined_baseline(
                self.result_dict.get("unbinned_baseline"))

            if len(self.datasets) > 1:
                combined_block = self._format_sigma_block(
                    "combined",
                    combined_val_sigmas, combined_train_sigmas,
                    combined_val_loss, combined_train_loss,
                    combined_std, combined_unb,
                )

        # --- Persist combined scalars (use setdefault for backward compat
        # with result_dicts loaded from before this change). ---
        self.result_dict.setdefault("combined_val_loss", []).append(
            combined_val_sigmas if combined_val_sigmas is not None else {})
        self.result_dict.setdefault("combined_val_loss_scalar", []).append(
            float(combined_val_loss))
        self.result_dict.setdefault("combined_train_val_loss_scalar", []).append(
            float(combined_train_loss))

        # --- Print everything in one tidy block ---
        print("\n=== Validation ===")
        for b in blocks:
            print(b)
            print("")
        if combined_block is not None:
            print(combined_block)
            best_so_far = self.result_dict["best_val_loss"]
            best_str = (f"{float(best_so_far):.6f}"
                        if best_so_far != jnp.inf else "n/a")
            print(f"           best so far: {best_str}")
            print("")

        # --- Best-params tracking: lowest combined val loss wins. ---
        # With one dataset, "combined" equals that dataset's val loss.
        # combined_val_sigmas is only None if there were no MC datasets at
        # all, which would already have skipped the rest of validate().
        if combined_val_sigmas is not None:
            candidate = float(combined_val_loss)
            if candidate < self.result_dict["best_val_loss"]:
                self.result_dict["best_val_loss"] = candidate
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
