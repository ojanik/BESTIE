import jax.numpy as jnp
from jax import hessian
from .utilities import parse_yaml
from .nets import net_handler

class Pipeline():
    def __init__(self,config_path):
        self.config = parse_yaml(config_path)
        self.pipeline = None

        self.data = None
        self.aux = None
        self.injected_params = None

        self.calc_weights = None
        model = net_handler(self.config)
        self.net = model()
        self.calc_data_hist = None
        self.calc_hist = None
        self.calc_loss = None
        self.calc_llh = None

    def get_pipeline(self, rebuild = False):
        if self.pipeline == None or rebuild:
            self.pipeline = self.set_pipeline()

        return self.pipeline

    def set_pipeline(self):
        def analysis_llh(injected_params,lss,aux,data_hist):
            weights = self.calc_weights(injected_params,aux)
            hist = self.calc_hist(lss,weights)
            llh = self.calc_llh(hist,data_hist)

            return llh
        
        def pipeline(params):
            lss = self.net.apply({"params": params},self.data)
            data_hist = self.calc_data_hist(self.injected_params,self.data,self.aux)

            fish = hessian(analysis_llh)(self.injected_params,lss,self.aux,data_hist)
            CovMat = jnp.linalg.solve(fish)

            loss = self.calc_loss(CovMat)

            return loss

        self.pipeline = pipeline

if __name__ == "__main__":
    print("This is a module meant for importing only, NOT a script that can be executed!")


