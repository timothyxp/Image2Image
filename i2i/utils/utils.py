import numpy as np


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def set_require_grad(model, value: bool):
    for param in model.parameters():
        param.requires_grad = value
