import optax
import jax.numpy as jnp


def create_cosine_lr(base_learning_rate, steps_per_epoch=1,num_epochs = 1000,warmup_epochs = 10):
    warmup_fn = optax.linear_schedule(
      init_value=0., end_value=base_learning_rate,
      transition_steps=warmup_epochs * steps_per_epoch)
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
      init_value=base_learning_rate,
      decay_steps=cosine_epochs * steps_per_epoch)
    schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[warmup_epochs * steps_per_epoch])
    return schedule_fn

def create_cyclic_lr(max_lr,min_lr,number_of_epochs,number_of_periods):
  if max_lr < min_lr:
          raise ValueError("min_lr cannot be larger than max_lr")
  def cyclic_schedule_fn(x):
      diff = max_lr - min_lr
      T = number_of_epochs / number_of_periods
      return 10**((diff/2)*(jnp.cos(jnp.pi*(2*x/T))+1)+min_lr)
  return cyclic_schedule_fn