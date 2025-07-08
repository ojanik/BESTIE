from .hists import hist_handler
from .transformations import transformation_handler
from .nets import model_handler
from .losses import loss_handler

from functools import partial

from jax import jit
import jax.numpy as jnp
import jax
Array = jnp.array


class Pipeline():
    def __init__(self,config):
        self.config = config

        self.calc_hist = hist_handler(self.config["hists"])
        self.transform_fun = transformation_handler(self.config["transformation"])
        self.model = model_handler(self.config)
        self.net = self.model()

        self.calc_loss = loss_handler(self.config)
        self._set_optimization_pipeline()

    def calc_lss(self, net_params,data,training=True,drop_out_key=None):
        lss = self.net.apply({"params":net_params},data,training=training,rngs={'dropout': drop_out_key})
        if self.config["hists"]["method"].lower() != "vector":
            lss = self.transform_fun(lss)
        lss *= self.config["hists"]["bins_up"]
        return lss

    def _set_optimization_pipeline(self):
        @partial(jit,static_argnames=["training"])
        def optimization_pipeline(net_params,data,weights,grad_weights,sample_weights,training=True,drop_out_key=None):
            lss = self.calc_lss(net_params,data,drop_out_key=drop_out_key,training=training)
            all_weights = self.calc_hist(lss)
            #jax.debug.print("{x}",x=all_weights.sum()/self.config["training"]["batch_size"])
            mu = self.get_hist(all_weights,weights,sample_weights)
            ssq = self.get_hist(all_weights,weights**2,sample_weights)
            grad_hist = jax.tree_util.tree_map(lambda v: self.get_hist(all_weights, v, sample_weights), grad_weights)
            
            loss = self.calc_loss(mu,ssq,grad_hist)

            return jnp.sum(loss), loss
        self._optimization_pipeline = optimization_pipeline

    def test_hist(self, net_params,data,weights,sample_weights,rng):
        lss = self.calc_lss(net_params,data,drop_out_key=rng,training=False)
        all_weights = self.calc_hist(lss)
        
        mu = self.get_hist(all_weights,weights,sample_weights)
        return mu 

    def get_hist(self,all_weights,weights,sample_weights=None):

        if sample_weights is not None:
                sample_weights = jnp.reshape(sample_weights,weights.shape)
                weights = weights * sample_weights

        counts = jnp.sum(all_weights * weights[:, None], axis=0)

        return counts
    
    def eval_hists(self,net_params,data,weights,grad_weights,sample_weights,training=False,drop_out_key=None):
        lss = self.calc_lss(net_params,data,drop_out_key=drop_out_key,training=training)
        mu = self.get_hist(lss,weights,sample_weights)
        ssq = self.get_hist(lss,weights**2,sample_weights)
        grad_hist = jax.tree_util.tree_map(lambda v: self.get_hist(lss, v, sample_weights), grad_weights)
        return mu,ssq,grad_hist

if __name__ == "__main__":
    print("This is a module meant for importing only, NOT a script that can be executed!")


