import torch
import torch.nn as nn


class Scale(nn.Module):

    def __init__(self, nchannels, bias=True, init_scale=1.0):
        super().__init__()
        self.nchannels = nchannels
        self.weight = nn.Parameter(torch.Tensor(1, nchannels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, nchannels, 1, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(init_scale)

    def reset_parameters(self, init_scale=1.0):
        self.weight.data.fill_(init_scale)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x):
        y = x * self.weight
        if self.bias is not None:
            y += self.bias
        return y

    def __repr__(self):
        s = '{} ({}, {})'
        return s.format(self.__class__.__name__, self.nchannels, self.bias
             is not None)


class L2Norm(nn.Module):

    def __init__(self, nchannels, bias=True):
        super().__init__()
        self.scale = Scale(nchannels, bias=bias)
        self.nchannels = nchannels
        self.eps = 1e-06

    def forward(self, x):
        l2_norm = x.norm(2, dim=1, keepdim=True) + self.eps
        x_norm = x.div(l2_norm)
        y = self.scale(x_norm)
        return y

    def __repr__(self):
        s = '{name} ({nchannels})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nchannels': 4}]
