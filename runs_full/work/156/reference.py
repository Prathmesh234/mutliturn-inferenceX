import torch
from torch import nn


class HSwish(nn.Module):
    """Hard Swish activation function.
    See: https://arxiv.org/abs/1905.02244
    """

    def forward(self, x):
        return x * nn.functional.relu6(x + 3).div_(6)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
