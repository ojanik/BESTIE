from src import BESTIE
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

config_path = "./src/BESTIE/configs/general_config.yaml"

Pipe = BESTIE.AnalysisPipeline(config_path)