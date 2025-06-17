# to be deleted
"""from torch.utils.data import DataLoader
from tqdm import tqdm
import jax.numpy as jnp
Array = jnp.array
from jax import value_and_grad, jit
from .nets import lr_handler
import optax
from flax.training import train_state

def calc_bin_idx(data):
    energy = Array(data[:,0])
    coszenith = jnp.cos(Array(data[:,1]))

    energy_bins = jnp.logspace(2,7,51)
    coszenith_bins = jnp.linspace(-1,0.0872,34)

    energy_digi = jnp.digitize(energy,energy_bins) - 1
    zenith_digi = jnp.digitize(coszenith,coszenith_bins) - 1
    bins_flattened = energy_digi * 33 + zenith_digi

    return bins_flattened





def train_shape(net,init_params,dl,config):
    batch_size = config["training"]["batch_size"]
    #batch_size = 64

    def shape_loss(params,input_data,data):
        preds = net.apply({"params":params},input_data)[:,0]
        preds -= jnp.min(preds)
        preds /= jnp.max(preds)
        bin_idx = calc_bin_idx(data)
        truth = bin_idx / config["hists"]["bins_number"]
        return jnp.sum((preds-truth)**2) / len(preds)

    calc_loss = jit(shape_loss)
    steps_per_epoch = len(dl)
    config["training"]["lr"]["lr"] = 0.01
    lr = lr_handler(config,steps_per_epoch)

    tx = getattr(optax,config["training"]["optimizer"].lower())(learning_rate = lr)

    state = train_state.TrainState.create(apply_fn=net.apply,
                                          params=init_params,
                                          tx=tx)
    for j in (tpbar:= tqdm(range(5))):
        running_loss = 0
        pbar = tqdm(enumerate(dl), total=len(dl))
        for i,(data,aux,real_data,_) in pbar:
            data = Array(data)
            real_data = Array(real_data)

            for key in aux.keys():
                aux[key] = Array(aux[key])

            #data_hist = asimov_func(state.params,Array(list(injected_params.values())),data,aux)
            #for k in range(50):
            
            loss, grads = value_and_grad(calc_loss)(state.params,data,real_data)
            state = state.apply_gradients(grads=grads)

            pbar.set_description(f"loss: {loss:.9f}")
            running_loss += loss
        avg_loss = running_loss / len(dl)
        tpbar.set_description(f"loss: {avg_loss:.9f}")
    
    return state.params"""