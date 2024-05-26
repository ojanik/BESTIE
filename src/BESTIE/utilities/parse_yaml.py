import yaml

def parse_yaml(config_file:str):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config
