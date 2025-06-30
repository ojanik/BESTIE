from .hists import hist_handler
from .transformations import transformation_handler

from functools import partial

from jax import jit
import jax.numpy as jnp
import jax
Array = jnp.array

from tqdm import tqdm

import BESTIE

class AnalysisPipeline():
    def __init__(self,config):
        self.config = config
        injected_params = config["injected_params"].copy()
        for key in injected_params.keys():
            injected_params[key] = Array(injected_params[key])
        self.injected_parameter_keys =list(injected_params.keys())

        self._analysis_pipeline = None
        self.calc_weights = weight_handler(self.config)
        self.calc_hist = hist_handler(self.config["hists"])
        self.calc_llh = llh_handler(self.config["llh"])
        self.transform_fun = transformation_handler(self.config["transformation"])


    def get_analysis_pipeline(self, rebuild = False):
        if self._analysis_pipeline == None or rebuild:
            self._set_analysis_pipeline()

        return self._analysis_pipeline

    def _set_analysis_pipeline(self):

        @partial(jax.jit, static_argnames=['skip_llh','opti'])
        def analysis_pipeline(injected_params_values,lss,aux,data_hist,sample_weights=None,skip_llh=False,only_sample_weights=False,**kwargs):
            mu, sigma = self.get_hist(lss,injected_params_values,aux,sample_weights,only_sample_weights)
            #mu = mu.sum(axis=0) +1e-6
            #sigma = sigma.sum(axis=0) +1e-6
            if skip_llh:
                return mu, sigma
            llh = self.calc_llh(data_hist,mu) 
            llh = llh.sum() / self.config["hists"]["bins_number"]
            return llh, sigma
        self._analysis_pipeline = analysis_pipeline

    def get_hist(self,lss,injected_params,aux,sample_weights=None,only_sample_weights=False):
        injected_params = dict(zip(self.injected_parameter_keys,injected_params))
        weights = self.calc_weights(injected_params,aux)
        if sample_weights is not None:
                sample_weights = jnp.reshape(sample_weights,weights.shape)
                mu_weights = weights * sample_weights
                ssq_weights = weights**2 * sample_weights
        if only_sample_weights:
            weights = sample_weights
        mu, sigma = self.calc_hist(lss, mu_weights=mu_weights,ssq_weights= ssq_weights)

        return mu, sigma

from .nets import model_handler
from .losses import loss_handler

class Optimization_Pipeline(AnalysisPipeline):
    def __init__(self,config,):
        super().__init__(config)

        self._optimization_pipeline = None

        self.model = model_handler(self.config)
        self.net = self.model()

        self.calc_loss = loss_handler(self.config)

        self._set_analysis_pipeline()


    def get_optimization_pipeline(self, rebuild = False):
        if self._optimization_pipeline == None or rebuild:
            self._set_optimization_pipeline()

        return self._optimization_pipeline

    def _set_optimization_pipeline(self):
        @partial(jit,static_argnames=["training"])
        def optimization_pipeline(net_params,injected_params,data,aux,sample_weights,**kwargs):
            lss = self.calc_lss(net_params,data,**kwargs)
            #jax.debug.print("{x}",x=lss)
            data_hist, _ = self.get_hist(lss,injected_params,aux,sample_weights)
            #data_hist = self.data_hist
            loss = self.calc_loss(self._analysis_pipeline,injected_params,lss,aux,data_hist,sample_weights,**kwargs)

            return jnp.sum(loss), loss
        self._optimization_pipeline = optimization_pipeline

    def calc_lss(self, net_params,data,training=True,drop_out_key=None,**kwargs):
        lss = self.net.apply({"params":net_params},data,training=training,rngs={'dropout': drop_out_key})
        if self.config["hists"]["method"].lower() != "vector":
            lss = self.transform_fun(lss,**kwargs)
        lss *= self.config["hists"]["bins_up"]
        if "scale" in net_params:
            lss *= jax.nn.sigmoid(net_params["scale"])
            lss *= 2
        return lss
    
    def get_loss(self,net_params,injected_params,data,aux,data_hist):
        lss = self.net.apply({"params":net_params},data)[:,0]
        loss = self.calc_loss(self._analysis_pipeline,injected_params,lss,aux,data_hist)
        
        return loss
    
    def get_asimovhist_func(self):
        def calc_asimovhist(net_params,injected_params,data,aux,**kwargs):
            lss = self.get_lss(net_params,data)
            
            lss = self.transform_fun(lss,**kwargs)
            injected_params = dict(zip(self.injected_parameter_keys,injected_params))
            weights = self.calc_weights(injected_params,aux)
            hist = self.calc_hist(lss, weights=weights)

            return hist
        
        return calc_asimovhist

    def calc_data_hist(self, net_params, data, aux, injected_params, batch_size = 10000):
        num_parts = int(jnp.ceil(len(data)/batch_size))
        apply_fn = jit(self.net.apply)
        print("--- Calculating lss ---")
        for i in tqdm(range(num_parts),disable=True):
            batched_input_data = data[i*batch_size:jnp.min(Array([(i+1)*batch_size,len(data)])),:-1]
            batched_input_data = BESTIE.data.fourier_feature_mapping.input_mapping(batched_input_data,B)
            if i == 0:
                lss = apply_fn({"params": net_params},batched_input_data)[:,0]
            else:
                lss = jnp.concatenate([lss,apply_fn({"params": net_params},batched_input_data)[:,0]])

        weights = self.calc_weights(injected_params,aux)
        self.data_hist = jnp.histogram(lss,
                                       weights = weights, 
                                       bins=jnp.linspace(self.config["hists"]["bins_low"],
                                                             self.config["hists"]["bins_up"],
                                                             self.config["hists"]["bins_number"]))

if __name__ == "__main__":
    print("This is a module meant for importing only, NOT a script that can be executed!")


