from .utilities import parse_yaml
from .nets import model_handler
from .llh import llh_handler

class AnalysisPipeline():
    def __init__(self,config_path):
        self.config = parse_yaml(config_path)
        self._pipeline = None

        self.calc_data_hist = None
        self.calc_hist = None
        self.calc_llh = llh_handler(self.config)

    def get_analysis_pipeline(self, rebuild = False):
        if self._pipeline == None or rebuild:
            self._analysis_pipeline = self._set_analysis_pipeline()

        return self._pipeline

    def _set_analysis_pipeline(self):
        def analysis_pipeline(injected_params,lss,aux,data_hist):
            weights = self.calc_weights(injected_params,aux)
            hist = self.calc_hist(lss,weights)
            llh = self.calc_llh(hist,data_hist)

            return llh
        self._analysis_pipeline = analysis_pipeline
 
if __name__ == "__main__":
    print("This is a module meant for importing only, NOT a script that can be executed!")


