import torch
from torch import nn


class Mish(nn.Module):
    """Mish activation."""

    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
