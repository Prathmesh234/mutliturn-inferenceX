import torch
from torch import nn


class ScaleNorm(nn.Module):

    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.scale = dim ** -0.5
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps
            ) * self.scale
        x = x / n * self.g
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
