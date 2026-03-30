import torch
from .registry import model_dict


def load_model(model_name, model_path=None):

    if model_name not in model_dict:
        raise ValueError(f'Model {model_name} not found in model_dict')
    model_class = model_dict[model_name]
    model = model_class()
    if model_path is not None:
        weights = torch.load(model_path)
        model.load_state_dict(weights)
    return model

