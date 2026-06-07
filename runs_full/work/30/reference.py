import torch
import torch.nn as nn


class NormalizationLayer(nn.Module):

    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def forward(self, x, epsilon=1e-08):
        return x * ((x ** 2).mean(dim=1, keepdim=True) + epsilon).rsqrt()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
