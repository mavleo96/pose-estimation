import json


def load_models_info(path):
    with open(path) as f:
        models_info = json.load(f)
        models_info = {int(k): v for k, v in models_info.items()}
    return models_info


__all__ = ["load_models_info"]
