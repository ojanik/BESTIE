from .utilities import parse_yaml
from .llh import llh_handler
from .weights import weight_handler
from .hists import hist_handler


from jax import jit

class AnalysisPipeline():
    def __init__(self,config_path):
        self.config = parse_yaml(config_path)
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
        def analysis_pipeline(injected_params,lss,aux,data_hist):
            weights = self.calc_weights(injected_params,aux)
            hist = self.calc_hist(lss,weights=weights)
            llh = self.calc_llh(hist,data_hist)

            return llh
        self._analysis_pipeline = analysis_pipeline
 
if __name__ == "__main__":
    print("This is a module meant for importing only, NOT a script that can be executed!")


