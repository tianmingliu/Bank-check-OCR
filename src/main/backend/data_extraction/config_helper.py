import yaml
import os

"""
This method retrieves the configuration data that exists in the config.yml file

@return cfg the configuration data
"""


def get_config_data():
    filedir = os.path.abspath(os.path.dirname(__file__))
    filename = os.path.join(filedir, '../../config/config.yml')
    with open(filename, 'r') as config_file:
        cfg = yaml.safe_load(config_file)

    return cfg
