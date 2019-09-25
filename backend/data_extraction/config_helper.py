import yaml
import os


def get_config_data():
    filedir = os.path.dirname(os.path.realpath('__file__'))
    filename = os.path.join(filedir, '../../../../config/config.yml')

    with open(filename, 'r') as config_file:
        cfg = yaml.load(config_file)

    return cfg
