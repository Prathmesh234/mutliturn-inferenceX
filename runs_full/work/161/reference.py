import torch
import torch.nn as nn


class ClippedReLU(nn.Module):

    def __init__(self):
        super(ClippedReLU, self).__init__()

    def forward(self, x):
        return x.clamp(min=0.0, max=255.0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
