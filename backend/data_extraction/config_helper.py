import yaml
import os


def get_config_data():
    filedir = os.path.abspath(os.path.dirname(__file__))
    print(filedir)
    filename = os.path.join(filedir, '..\\..\\config\\config.yml')
    print(filename)
    with open(filename, 'r') as config_file:
        cfg = yaml.load(config_file)

    return cfg