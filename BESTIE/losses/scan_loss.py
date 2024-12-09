from jax.tree_util import Partial
from jaxopt import LBFGS

import jax




def get_partial_fun(ana_fun,lss,aux,data_hist,sample_weights=None):
    partial_fun = Partial(ana_fun,
                          lss=lss,
                          aux=aux,
                          data_hist=data_hist,
                          sample_weight=sample_weights)
    return partial_fun

def freefit(ana_fun,injected_params_values,lss,aux,data_hist,sample_weights):
    solver = LBFGS(fun=ana_fun, maxiter = 5000, implicit_diff=True)
    params, state = solver.run(injected_params_values,lss=lss,aux=aux,data_hist=data_hist,sample_weights=sample_weights)
    jax.debug.print("Freefit done: {x}",x=params)
    return params

def get_scan_fit(ana_fun,injected_params_values,lss,aux,data_hist,sample_weights,scan_parameter_idx,scan_parameter_value):

    def fixed_parameter_fun(injected_params_values,lss,aux,data_hist,sample_weights,scan_parameter_idx,scan_parameter_value):
        params = injected_params_values.at[scan_parameter_idx].set(scan_parameter_value) #remove index assignment

        return ana_fun(params,lss,aux,data_hist,sample_weights)

    solver = LBFGS(fun=fixed_parameter_fun, maxiter = 5000, implicit_diff=True)
    params, state = solver.run(injected_params_values,
                               lss=lss,
                               aux=aux,
                               data_hist=data_hist,
                               sample_weights=sample_weights,
                               scan_parameter_idx=scan_parameter_idx,
                               scan_parameter_value=scan_parameter_value)

    return params

def calc_scan_loss(ana_fun,injected_params_values,lss,aux,data_hist,sample_weights=None,scan_parameter_idx=0,**kwargs):
    #partial_fun = get_partial_fun(ana_fun,lss,aux,data_hist,sample_weights)

    free_fit_params = freefit(ana_fun,injected_params_values,lss,aux,data_hist,sample_weights)

    signal_fit = free_fit_params[scan_parameter_idx]

    up, low = signal_fit * 0.9 , signal_fit*1.1

    up_params = get_scan_fit(ana_fun,injected_params_values,lss,aux,data_hist,sample_weights,scan_parameter_idx, up)
    low_params = get_scan_fit(ana_fun,injected_params_values,lss,aux,data_hist,sample_weights,scan_parameter_idx, low)

    llh_low = ana_fun(low_params,lss,aux,data_hist,sample_weights)
    llh_up = ana_fun(up_params,lss,aux,data_hist,sample_weights)


    jax.debug.print("Debug: llh values: {x}, {y}",x=llh_low,y=llh_up)
    
    return llh_low+llh_up

#write test function
#write scan fits


