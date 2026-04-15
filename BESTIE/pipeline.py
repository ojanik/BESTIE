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

        # Ensure this returns a dict with an 'apply' function
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

    def calc_lss(self,net_params,data,hist_map,name,training,drop_out_key):
        hkey = hist_map[name]

        lss = self.net_dict[hkey]["net"].apply(
            net_params[hkey],
            data,
            training=training,
            rngs={"dropout": drop_out_key}
            )

        #lss *= self.config["hists"][hkey]["hists"]["bins_up"]
        lss = self.transform_fun(lss)
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

    def _set_optimization_pipeline(self, hist_map: dict):
        @partial(jax.jit, static_argnames=["training"])
        def optimization_pipeline(net_params, data_dict, training=True, drop_out_key=None):
            lss_dict = self.calc_lss_dict(net_params, data_dict, hist_map,
                                     training=training, drop_out_key=drop_out_key)

            hist_names = {k: hist_map[k] for k in lss_dict}
            
            hist_dict = self.get_histograms(lss_dict, hist_names)

            grouped = {}
            for name, entry in hist_dict.items():
                hname = hist_names[name]
                if hname not in grouped:
                    grouped[hname] = {
                        "mu": entry["mu"],
                        "ssq": entry["ssq"],
                        "grad_hist": entry["grad_hist"].copy()
                    }
                else:
                    grouped[hname]["mu"] += entry["mu"]
                    grouped[hname]["ssq"] += entry["ssq"]
                    for k, v in entry["grad_hist"].items():
                        if k in grouped[hname]["grad_hist"]:
                            grouped[hname]["grad_hist"][k] += v
                        else:
                            grouped[hname]["grad_hist"][k] = v

            all_mu = []
            all_ssq = []
            grad_chunks = []
            all_keys = set()

            for group in grouped.values():
                all_mu.append(group["mu"])
                all_ssq.append(group["ssq"])
                grad_chunks.append(group["grad_hist"])
                all_keys.update(group["grad_hist"].keys())

            chunk_lengths = [g["mu"].shape[0] for g in grouped.values()]

            grad_hist = {}
            for k in all_keys:
                vs = []
                for chunk, length in zip(grad_chunks, chunk_lengths):
                    if k in chunk:
                        vs.append(chunk[k])
                    else:
                        example = next(iter(chunk.values()))
                        shape = (length,) + example.shape[1:]
                        vs.append(jnp.zeros(shape, dtype=example.dtype))
                grad_hist[k] = jnp.concatenate(vs)

            mu = jnp.concatenate(all_mu)
            #("Mu sum {x}",x=mu.sum())
            ssq = jnp.concatenate(all_ssq)
            losses = self.calc_loss(mu, ssq, grad_hist)
            total_loss = jnp.sum(losses)

            return total_loss, losses

        self._optimization_pipeline = optimization_pipeline

    def test_hist(self, net_params, data_dict, rng):
        lss_dict = self.calc_lss_dict(net_params, data_dict, self.hist_map,
                                 drop_out_key=rng, training=False)
        hist_names = {k: self.hist_map[k] for k in data_dict}
        hist_dict = self.get_histograms(lss_dict, hist_names)

        grouped = {}
        for name, entry in hist_dict.items():
            hname = hist_names[name]
            if hname not in grouped:
                grouped[hname] = {
                    "mu": entry["mu"],
                    "ssq": entry["ssq"],
                    "grad_hist": entry["grad_hist"].copy()
                }
            else:
                grouped[hname]["mu"] += entry["mu"]
                grouped[hname]["ssq"] += entry["ssq"]
                for k, v in entry["grad_hist"].items():
                    if k in grouped[hname]["grad_hist"]:
                        grouped[hname]["grad_hist"][k] += v
                    else:
                        grouped[hname]["grad_hist"][k] = v

        all_mu = []
        all_ssq = []
        grad_chunks = []
        all_keys = set()

        for group in grouped.values():
            all_mu.append(group["mu"])
            all_ssq.append(group["ssq"])
            grad_chunks.append(group["grad_hist"])
            all_keys.update(group["grad_hist"].keys())

        chunk_lengths = [g["mu"].shape[0] for g in grouped.values()]

        grad_hist = {}
        for k in all_keys:
            vs = []
            for chunk, length in zip(grad_chunks, chunk_lengths):
                if k in chunk:
                    vs.append(chunk[k])
                else:
                    example = next(iter(chunk.values()))
                    shape = (length,) + example.shape[1:]
                    vs.append(jnp.zeros(shape, dtype=example.dtype))
            grad_hist[k] = jnp.concatenate(vs)

        # clip_mask = mu < 1e-2



        mu = jnp.concatenate(all_mu)
        
        ssq = jnp.concatenate(all_ssq)

        return mu, ssq, grad_hist



    def eval_hists(self, net_params, data_dict, training=False, drop_out_key=None):
        lss_dict = self.calc_lss(net_params, data_dict, self.hist_map,
                                 drop_out_key=drop_out_key, training=training)
        hist_names = {k: self.hist_map[k] for k in data_dict}
        return self.get_histograms(lss_dict, hist_names)


if __name__ == "__main__":
    print("This is a module meant for importing only, NOT a script that can be executed!")