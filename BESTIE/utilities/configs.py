import yaml
import collections.abc

def parse_yaml(config_file:str):
    """
    Parses a YAML configuration file and returns the configuration as a dictionary.

    Args:
        config_file (str): The path to the YAML configuration file.

    Returns:
        dict: The parsed configuration as a dictionary.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def override(d, u):
    """
    Recursively overrides the values in dictionary `d` with the values from dictionary `u`.

    Args:
        d (dict): The dictionary to be overridden.
        u (dict): The dictionary containing the values to override with.

    Returns:
        dict: The overridden dictionary.

    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = override(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def write_yaml(config_dict, path):
    """
    Write a YAML file with the given configuration dictionary.

    Args:
        config_dict (dict): The dictionary containing the configuration data.
        path (str): The path to the output YAML file.

    Returns:
        None
    """
    with open(path, 'w') as file:
        documents = yaml.dump(config_dict, file)
