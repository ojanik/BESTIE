from .utilities import parse_yaml
from .llh import llh_handler
from .weights import weight_handler
from .hists import hist_handler

from jax import jit
import jax.numpy as jnp
Array = jnp.array

class AnalysisPipeline():
    def __init__(self,config,injected_parameter_keys):
        #self.config = parse_yaml(config_path)
        self.config = config
        self.injected_parameter_keys = injected_parameter_keys

        self._analysis_pipeline = None
        self.calc_weights = weight_handler(self.config["weights"])
        self.calc_hist = hist_handler(self.config["hists"])
        self.calc_llh = llh_handler(self.config["llh"])

    def get_analysis_pipeline(self, rebuild = False):
        if self._analysis_pipeline == None or rebuild:
            self._set_analysis_pipeline()

        return self._analysis_pipeline

    def _set_analysis_pipeline(self):

        @jit
        def analysis_pipeline(injected_params_values,lss,aux,data_hist):
            injected_params = dict(zip(self.injected_parameter_keys,injected_params_values))
            weights = self.calc_weights(injected_params,aux)
            hist = self.calc_hist(lss,weights=weights)
            llh = self.calc_llh(data_hist,hist)
            llh = llh.sum() / self.config["hists"]["bins_number"]
            return llh
        self._analysis_pipeline = analysis_pipeline


from .nets import model_handler
from .losses import loss_handler

class Optimization_Pipeline(AnalysisPipeline):
    def __init__(self,config,injected_parameter_keys):
        super().__init__(config,injected_parameter_keys)

        self._optimization_pipeline = None

        self.model = model_handler(self.config)
        self.net = self.model()

        self.calc_loss = loss_handler(self.config["loss"])

        self._set_analysis_pipeline()


    def get_optimization_pipeline(self, rebuild = False):
        if self._optimization_pipeline == None or rebuild:
            self._set_optimization_pipeline()

        return self._optimization_pipeline

    def _set_optimization_pipeline(self):
        @jit
        def optimization_pipeline(net_params,injected_params,data,aux):
            lss = self.net.apply({"params":net_params},data)[:,0]
            #if self.config["network"]["hidden_layers"][-1]["activation"].lower() == "lin":
            lss -= jnp.min(lss)
            lss /= jnp.max(lss)
            data_hist = self.get_hist(lss,injected_params,aux)
            loss = self.calc_loss(self._analysis_pipeline,injected_params,lss,aux,data_hist)

            return loss
        self._optimization_pipeline = optimization_pipeline

    def get_lss(self, net_params,data):
        lss = self.net.apply({"params":net_params},data)[:,0]
        return lss
    
    def get_loss(self,net_params,injected_params,data,aux,data_hist):
        lss = self.net.apply({"params":net_params},data)[:,0]
        loss = self.calc_loss(self._analysis_pipeline,injected_params,lss,aux,data_hist)

        return loss


    def get_hist(self,lss,injected_params,aux):
        injected_params = dict(zip(self.injected_parameter_keys,injected_params))
        weights = self.calc_weights(injected_params,aux)
        hist = self.calc_hist(lss, weights=weights)

        return hist
    
    def get_asimovhist_func(self):
        def calc_asimovhist(net_params,injected_params,data,aux):
            lss = self.get_lss(net_params,data)
            injected_params = dict(zip(self.injected_parameter_keys,injected_params))
            weights = self.calc_weights(injected_params,aux)
            hist = self.calc_hist(lss, weights=weights)

            return hist
        
        return calc_asimovhist

if __name__ == "__main__":
    print("This is a module meant for importing only, NOT a script that can be executed!")


