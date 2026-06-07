import torch
from torch import nn


class GlobalAveragePool(nn.Module):
    """
    Average pooling in an equivariant network
    """

    def __init__(self):
        """
        """
        super().__init__()

    def forward(self, x):
        """
        """
        avg = torch.mean(x, dim=[-2, -1], keepdim=True)
        return avg


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
