import torch
import torch.utils.data
import torch.nn as nn


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-06, gamma=1.0, beta=0.0, learnable=
        False):
        super(LayerNorm, self).__init__()
        if learnable:
            self.gamma = nn.Parameter(torch.ones(features))
            self.beta = nn.Parameter(torch.zeros(features))
        else:
            self.gamma = gamma
            self.beta = beta
        self.eps = eps

    def forward(self, x):
        x_size = x.size()
        mean = x.view(x_size[0], x_size[1], x_size[2] * x_size[3]).mean(2
            ).view(x_size[0], x_size[1], 1, 1).repeat(1, 1, x_size[2],
            x_size[3])
        std = x.view(x_size[0], x_size[1], x_size[2] * x_size[3]).std(2).view(
            x_size[0], x_size[1], 1, 1).repeat(1, 1, x_size[2], x_size[3])
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'features': 4}]
