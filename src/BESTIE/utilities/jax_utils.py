from jax.tree_util import tree_map

## A collection of functions acting on pytrees

def add_pytrees(pytree1, pytree2):
  return tree_map(lambda pt1, pt2: pt1 + pt2, pytree1, pytree2)

def substract_pytrees(pytree1, pytree2):
  return tree_map(lambda pt1, pt2: pt1 - pt2, pytree1, pytree2)

def multiply_pytrees(pytree1, pytree2):
  return tree_map(lambda pt1, pt2: pt1 * pt2, pytree1, pytree2)

def divide_pytrees(pytree1, pytree2):
  return tree_map(lambda pt1, pt2: pt1 / pt2, pytree1, pytree2)

def scale_pytrees(scalar, pytree):
  return tree_map(lambda pt: scalar * pt, pytree)