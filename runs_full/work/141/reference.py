import torch
import torch.nn as nn
import torch.nn


class MinPool(nn.Module):
    """Use nn.MaxPool to implement MinPool
    """

    def __init__(self, kernel_size, ndim=3, stride=None, padding=0,
        dilation=1, return_indices=False, ceil_mode=False):
        super(MinPool, self).__init__()
        self.pool = getattr(nn, f'MaxPool{ndim}d')(kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            return_indices=return_indices, ceil_mode=ceil_mode)

    def forward(self, x):
        x_max = x.max()
        x = self.pool(x_max - x)
        return x_max - x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
