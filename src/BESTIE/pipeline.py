import jax.numpy as jnp
from jax import hessian

class Pipeline():
    def __init__(config_path):
        self.config = parse_config(config_path)
        self.pipeline = None
    
    def get_pipeline():
        if self.pipeline == None:
            self.pipeline = self.set_pipeline()

        return self.pipeline

    def set_pipeline():
        def pipeline(params):
	    lss = net.apply({"params": params},data)
	    data_hist = calc_data_hist(injected_params,data,aux)

	    fish = hessian(analysis_llh)(injected_params,lss,aux,data_hist)
	    CovMat = jnp.linalg.solve(fish)

	    loss = calc_loss(CovMat)

	    return loss

        def analysis_llh(injected_params,lss,aux,data_hist):
	    weights = calc_weights(injected_params,aux)
	    hist = calc_hist(lss,weights)
	    llh = calc_llh(hist,data_hist)

	    return llh

        self.pipeline = pipeline



