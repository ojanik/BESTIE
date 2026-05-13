from .hists import hist_handler
from .transformations import transformation_handler
from .nets import model_handler
from .losses import loss_handler

from functools import partial
import jax
import jax.numpy as jnp


class Pipeline:
    def __init__(self, config):
        self.config = config

        # Parse the optional auxiliary score-regression head config. This MUST
        # run before `model_handler` is called, because it injects a
        # `score_head` sub-dict into each per-hist network config so the built
        # Flax module knows to instantiate the extra head and its parameters.
        self._init_score_head(config)

        self.calc_hist = hist_handler(self.config)
        self.transform_fun = transformation_handler(self.config["transformation"])
        self.model_dict = model_handler(self.config)

        self.net_dict = {
            hkey: {"net": self.model_dict[hkey]["model"]()}
            for hkey in self.model_dict
        }

        self.calc_loss = loss_handler(self.config)

        self.hist_map = {
            dkey: self.config["datasets"][dkey]["hist"]
            for dkey in self.config["datasets"]
        }

        self._set_optimization_pipeline(hist_map=self.hist_map)

    def _init_score_head(self, config):
        """Read loss.score_head and (if enabled) inject n_params into each
        per-hist network config so the built model adds the auxiliary head.

        Defaults preserve the previous behavior exactly: when `loss.score_head`
        is absent or `enabled` is False, no network or loss path changes.

        Optional decay schedule for the auxiliary loss weight:

            loss:
              score_head:
                enabled: true
                weight: 1.0            # initial weight
                decay:                  # optional sub-block; absent => constant weight
                  rate: 0.9             # per-epoch multiplicative decay
                  epochs: 50            # hard-zero the weight starting at this epoch

        With decay configured, the per-epoch weight is
            w(epoch) = weight * rate ** epoch        for epoch < decay.epochs
            w(epoch) = 0                              for epoch >= decay.epochs
        """
        sh_cfg = (config.get("loss", {}) or {}).get("score_head", {}) or {}
        self.score_head_enabled = bool(sh_cfg.get("enabled", False))

        if not self.score_head_enabled:
            self.score_params = []
            self.score_weight_initial = 0.0
            self.score_decay_rate = None
            self.score_decay_epochs = None
            self.score_eps = 1e-8
            return

        params = sh_cfg.get("parameters")
        if not params:
            raise ValueError(
                "loss.score_head.enabled=True requires loss.score_head.parameters "
                "to be a non-empty list of parameter names (matching keys in the "
                "dataset's grad_weights)."
            )
        # Sort to match Dataset.grad_weights ordering (alphabetical, see
        # dataset.py). This guarantees a stable column order in the score
        # target tensor.
        self.score_params = sorted(params)
        self.score_weight_initial = float(sh_cfg.get("weight", 1.0))
        self.score_eps = float(sh_cfg.get("eps", 1e-8))

        decay_cfg = sh_cfg.get("decay", {}) or {}
        rate = decay_cfg.get("rate", None)
        decay_epochs = decay_cfg.get("epochs", None)
        if rate is None and decay_epochs is None:
            # No decay configured — behaviour identical to the previous
            # constant-weight code path.
            self.score_decay_rate = None
            self.score_decay_epochs = None
        elif rate is None or decay_epochs is None:
            raise ValueError(
                "loss.score_head.decay requires BOTH 'rate' and 'epochs' to be set."
            )
        else:
            self.score_decay_rate = float(rate)
            self.score_decay_epochs = int(decay_epochs)

    def current_score_weight(self, epoch: int) -> float:
        """Return the auxiliary score-head loss weight for ``epoch``.

        - If the score head is disabled: 0.0 (the aux path is never taken).
        - If no decay schedule is configured: the constant initial weight.
        - Otherwise: ``weight * rate ** epoch`` for ``epoch < decay.epochs``,
          and 0.0 once we reach ``decay.epochs`` (exponential decay snapped to
          zero at the end of the configured span).
        """
        if not self.score_head_enabled:
            return 0.0
        if self.score_decay_rate is None:
            return self.score_weight_initial
        if epoch >= self.score_decay_epochs:
            return 0.0
        return self.score_weight_initial * (self.score_decay_rate ** epoch)

        n_params = len(self.score_params)
        hidden = sh_cfg.get("hidden", []) or []
        activation = sh_cfg.get("activation", "silu")

        # Inject into every hist's network config so build_jax_dense picks it
        # up. Mutates `config` in place; the trainer already mutates `config`
        # elsewhere (e.g. save_dir), so this is consistent with existing usage.
        for hkey in config["hists"].keys():
            net_cfg = config["hists"][hkey]["network"]
            net_cfg["score_head"] = {
                "enabled": True,
                "n_params": n_params,
                "hidden": hidden,
                "activation": activation,
            }

    def _calc_net_outputs(self, net_params, data, hkey, training, drop_out_key):
        """Single forward pass returning (lss, score_or_None).

        The Flax module returns a dict {"lss", "score"} when the score head is
        enabled and a plain LSS array otherwise; this helper normalises that
        so callers don't have to type-check.
        """
        out = self.net_dict[hkey]["net"].apply(
            net_params[hkey],
            data,
            training=training,
            rngs={"dropout": drop_out_key},
        )
        if isinstance(out, dict):
            lss = self.transform_fun(out["lss"])
            score = out.get("score", None)
        else:
            lss = self.transform_fun(out)
            score = None
        return lss, score

    def calc_lss(self,net_params,data,hist_map,name,training,drop_out_key):
        hkey = hist_map[name]
        lss, _ = self._calc_net_outputs(net_params, data, hkey, training, drop_out_key)
        return lss

    def calc_lss_dict(self, net_params: dict, data_dict: dict, hist_map: dict,
                 training: bool = True, drop_out_key: jax.Array = None) -> dict:
        """Compute LSS per dataset using the correct network via hist_map."""

        def _apply(name, entry):
            data = entry["data"]
            lss = self.calc_lss(net_params,data,hist_map,name,training,drop_out_key)
            
            return {
                "lss": lss,
                "weights": entry["weights"],
                "sample_weights": entry["sample_weights"],
                "grad_weights": entry["grad_weights"],
            }
        
        return {name: _apply(name, entry) for name, entry in data_dict.items()}

    def _get_histogram(self, all_weights, weights, sample_weights=None):
        if sample_weights is not None:
            weights = weights * jnp.reshape(sample_weights, weights.shape)
        return jnp.sum(all_weights * weights[:, None], axis=0)

    def get_histograms(self, lss_dict: dict, hist_names: dict) -> dict:
        """Compute histograms (mu, ssq, grad_hist, all_weights) for each dataset."""
        hist_dict = {}

        for name, entry in lss_dict.items():
            lss = entry["lss"]
            
            weights = entry["weights"]
            sample_weights = entry.get("sample_weights", None)
            grad_weights = entry["grad_weights"]

            all_weights = self.calc_hist[self.hist_map[name]]["calc_hist"](lss)

            mu = self._get_histogram(all_weights, weights, sample_weights)
            ssq = self._get_histogram(all_weights, weights**2, sample_weights)
            
            grad_hist = {
                k: self._get_histogram(all_weights, v, sample_weights)
                for k, v in grad_weights.items()
            }

            hist_dict[name] = {
                "all_weights": all_weights,
                "mu": mu,
                "ssq": ssq,
                "grad_hist": grad_hist,
            }

        return hist_dict

    def _group_and_concat_hists(self, hist_dict: dict, hist_names: dict):
        """Group per-dataset histograms by hist name, sum within each group, then
        concatenate across groups into single mu/ssq/grad_hist arrays."""
        grouped = {}
        for name, entry in hist_dict.items():
            hname = hist_names[name]
            if hname not in grouped:
                grouped[hname] = {
                    "mu": entry["mu"],
                    "ssq": entry["ssq"],
                    "grad_hist": entry["grad_hist"].copy(),
                }
            else:
                grouped[hname]["mu"] += entry["mu"]
                grouped[hname]["ssq"] += entry["ssq"]
                for k, v in entry["grad_hist"].items():
                    if k in grouped[hname]["grad_hist"]:
                        grouped[hname]["grad_hist"][k] += v
                    else:
                        grouped[hname]["grad_hist"][k] = v

        all_keys = set().union(*(g["grad_hist"].keys() for g in grouped.values()))
        chunk_lengths = [g["mu"].shape[0] for g in grouped.values()]

        grad_hist = {}
        for k in all_keys:
            vs = []
            for group, length in zip(grouped.values(), chunk_lengths):
                if k in group["grad_hist"]:
                    vs.append(group["grad_hist"][k])
                else:
                    example = next(iter(group["grad_hist"].values()))
                    vs.append(jnp.zeros((length,) + example.shape[1:], dtype=example.dtype))
            grad_hist[k] = jnp.concatenate(vs)

        mu = jnp.concatenate([g["mu"] for g in grouped.values()])
        ssq = jnp.concatenate([g["ssq"] for g in grouped.values()])
        return mu, ssq, grad_hist

    def _calc_lss_and_score_dict(self, net_params, data_dict, hist_map,
                                  training=True, drop_out_key=None):
        """Like calc_lss_dict but also returns per-dataset score predictions.

        Used internally by the optimisation pipeline so that we run the
        forward pass once and then split the outputs into the LSS path
        (binning + Fisher loss) and the score path (auxiliary regression).
        """
        lss_dict = {}
        score_dict = {}
        for name, entry in data_dict.items():
            hkey = hist_map[name]
            lss, score = self._calc_net_outputs(
                net_params, entry["data"], hkey, training, drop_out_key
            )
            lss_dict[name] = {
                "lss": lss,
                "weights": entry["weights"],
                "sample_weights": entry["sample_weights"],
                "grad_weights": entry["grad_weights"],
            }
            if score is not None:
                score_dict[name] = score
        return lss_dict, score_dict

    def _compute_score_loss(self, score_dict, data_dict):
        """Event-weighted MSE between the score head and grad_w / (w + eps).

        Loss = sum_i w_i * ||score_pred_i - grad_w_i / (w_i + eps)||^2
               -------------------------------------------------------
                                  sum_i w_i

        - Targets are stacked in the order given by `self.score_params`
          (sorted alphabetically — same order Dataset uses for grad_weights).
        - The per-event physical weight w_i is reused as the MSE weight so
          that the loss approximates an expectation under p(x|theta), which
          mirrors how the Fisher loss already consumes weights.
        - sample_weights (importance-sampling correction) are folded in the
          same way the histogram path does.
        """
        eps = self.score_eps
        num = jnp.array(0.0)
        den = jnp.array(0.0)
        for name, score_pred in score_dict.items():
            raw = data_dict[name]
            w = raw["weights"]                # (B,)
            sw = raw.get("sample_weights")    # (B,) or None
            grad_w = raw["grad_weights"]      # dict: param -> (B,)

            # Stack per-event score targets in the canonical parameter order.
            target = jnp.stack(
                [grad_w[p] / (w + eps) for p in self.score_params],
                axis=-1,
            )  # shape (B, n_params)

            per_event_sq = jnp.sum((score_pred - target) ** 2, axis=-1)  # (B,)

            event_w = w
            if sw is not None:
                event_w = event_w * jnp.reshape(sw, event_w.shape)

            num = num + jnp.sum(event_w * per_event_sq)
            den = den + jnp.sum(event_w)

        return num / (den + 1e-12)

    def _set_optimization_pipeline(self, hist_map: dict):
        @partial(jax.jit, static_argnames=["training"])
        def optimization_pipeline(net_params, data_dict, score_weight,
                                  training=True, drop_out_key=None):
            """Forward + loss. ``score_weight`` is a runtime JAX scalar that
            multiplies the auxiliary score-regression loss; the trainer is
            expected to pass in the current epoch's weight (see
            ``Pipeline.current_score_weight``). Because it is a runtime arg
            (not static), updating it between epochs does NOT trigger a
            recompile.
            """
            lss_dict, score_dict = self._calc_lss_and_score_dict(
                net_params, data_dict, hist_map,
                training=training, drop_out_key=drop_out_key,
            )
            hist_names = {k: hist_map[k] for k in lss_dict}
            hist_dict = self.get_histograms(lss_dict, hist_names)
            mu, ssq, grad_hist = self._group_and_concat_hists(hist_dict, hist_names)
            fisher_losses = self.calc_loss(mu, ssq, grad_hist)

            if self.score_head_enabled:
                aux_loss = self._compute_score_loss(score_dict, data_dict)
                total = jnp.sum(fisher_losses) + score_weight * aux_loss
                # Append aux loss to the per-loss array so it is logged
                # alongside the existing Fisher / bin losses without changing
                # the surrounding plumbing. The raw aux value (BEFORE the
                # score_weight multiplier) is logged so the curve is
                # independent of how the schedule weights it into the total.
                losses = jnp.concatenate([fisher_losses, jnp.array([aux_loss])])
                return total, losses

            return jnp.sum(fisher_losses), fisher_losses

        self._optimization_pipeline = optimization_pipeline

    def test_hist(self, net_params, data_dict, rng):
        """Return (mu, ssq, grad_hist) for the given data without running the full loss."""
        lss_dict = self.calc_lss_dict(net_params, data_dict, self.hist_map,
                                      drop_out_key=rng, training=False)
        hist_names = {k: self.hist_map[k] for k in data_dict}
        hist_dict = self.get_histograms(lss_dict, hist_names)
        return self._group_and_concat_hists(hist_dict, hist_names)



    def eval_hists(self, net_params, data_dict, training=False, drop_out_key=None):
        lss_dict = self.calc_lss(net_params, data_dict, self.hist_map,
                                 drop_out_key=drop_out_key, training=training)
        hist_names = {k: self.hist_map[k] for k in data_dict}
        return self.get_histograms(lss_dict, hist_names)


if __name__ == "__main__":
    print("This is a module meant for importing only, NOT a script that can be executed!")