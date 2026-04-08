import yaml
import os
def load_config(dataset_name, model_name):
    with open(f'configs/{dataset_name}/{model_name}.yaml', 'r') as file:
        return yaml.safe_load(file)
