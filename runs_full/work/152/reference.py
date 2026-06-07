import torch
from torch import nn


class GlobalMaxPool(nn.Module):
    """
    Max pooling in an equivariant network
    """

    def __init__(self):
        """
        """
        super().__init__()

    def forward(self, x):
        """
        """
        mx = torch.max(torch.max(x, dim=-1, keepdim=True)[0], dim=-2,
            keepdim=True)[0]
        return mx


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
