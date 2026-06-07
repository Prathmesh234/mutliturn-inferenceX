import torch
import torch.utils.data
import torch.nn as nn


class CaffeNormalize(nn.Module):

    def __init__(self, features, eps=1e-07):
        super(CaffeNormalize, self).__init__()
        self.scale = nn.Parameter(10.0 * torch.ones(features))
        self.eps = eps

    def forward(self, x):
        x_size = x.size()
        norm = x.norm(2, dim=1, keepdim=True)
        x = x.div(norm + self.eps)
        return x.mul(self.scale.view(1, x_size[1], 1, 1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'features': 4}]
