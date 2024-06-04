from NNMFit import AnalysisConfig
from NNMFit.utilities import EventwiseGraph
import aesara

class NNMFit_handler():

    def __init__(self,config):
        self.config_hdl = AnalysisConfig.from_configs(
            main_config_file=config["main_config"],
            analysis_config_file=config["analysis_config"],
            override_config_files=config["override_configs"],
            override_dict=None,
            config_dir=config["config_dir"],
        )

        self._weight_graph = None
        self._w_fn = None

    def _set_weight_graph(self):
        Graph = EventwiseGraph.from_configdict(self.config_hdl.to_dict())

        self._weight_graph = Graph.w_tensor['IC86_pass2_SnowStorm_example_tracks']
        

    def _set_weight_function(self):
        if self._weight_graph == None:
            NNMFit_handler._set_weight_graph(self)
        self._w_fn = aesara.function([], self._weight_graph,on_unused_input='warn',mode="JAX")

    def get_weight_function(self):
        if self._w_fn == None:
            NNMFit_handler._set_weight_function(self)
        return self._w_fn.vm.jit_fn
