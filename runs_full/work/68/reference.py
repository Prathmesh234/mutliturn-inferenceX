import torch
import torch.nn as nn


class Linear_sigmoid(nn.Module):

    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=bias)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4, 'dim_out': 4}]
