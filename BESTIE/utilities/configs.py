import yaml
import collections.abc

def parse_yaml(config_file:str):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def override(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = override(d.get(k, {}), v)
        else:
            d[k] = v
    return d