import torch
import torch.nn as nn


class LayerNorm(nn.Module):

    def __init__(self, weights, eps=1e-05):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(weights))
        self.beta = nn.Parameter(torch.zeros(weights))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'weights': 4}]
