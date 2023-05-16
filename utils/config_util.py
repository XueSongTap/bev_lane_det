import importlib


def load_config_module(config_file):
    spec = importlib.util.spec_from_file_location("tuly.config", config_file)
    configs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(configs)
    return configs