import numpy as np
import torch
from PIL import Image


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def set_require_grad(model, value: bool):
    for param in model.parameters():
        param.requires_grad = value


def convert_to_pil(img: torch.Tensor, scale=(0, 1)):
    img = img \
        .detach() \
        .clamp_(*scale) \
        .permute(1, 2, 0) \
        .cpu() \
        .numpy() \
        .dot(255) \
        .astype(np.uint8)

    img = Image.fromarray(img)

    return img
