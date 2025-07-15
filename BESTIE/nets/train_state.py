from flax.training import train_state
from typing import Any, Callable, Dict
import jax
import optax
from flax import struct


class MultiNetworkTrainState(train_state.TrainState):
    apply_fns: Dict[str, Callable] = struct.field(pytree_node=False)  # e.g., {'net1': Net1.apply, 'net2': Net2.apply}
    key: jax.Array                  # for RNG tracking (e.g., dropout)

    @classmethod
    def create(cls,
               apply_fns: Dict[str, Callable],
               params: Dict[str, Any],
               tx: optax.GradientTransformation,
               key: jax.Array):
        """Creates a new MultiNetworkTrainState with RNG support."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn = lambda *a, **kw: None, # dummy apply_fn
            apply_fns=apply_fns,
            params=params,
            tx=tx,
            opt_state=opt_state,
            key=key
        )