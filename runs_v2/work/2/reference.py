import torch
import torch.nn as nn


class CustomizeLayer(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.scale = nn.Parameter(torch.Tensor(self.in_dim))
        self.bias = nn.Parameter(torch.Tensor(self.in_dim))

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()
        x = torch.div(x, norm)
        return x * self.scale + self.bias

    def __repr__(self):
        return 'CustomizedLayer(in_dim=%d)' % self.in_dim


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4}]
