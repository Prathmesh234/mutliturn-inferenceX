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


class CReLU(nn.Module):

    def __init__(self, nchannels):
        super().__init__()
        self.scale = Scale(2 * nchannels)
        self.relu = nn.ReLU(inplace=True)
        self.in_channels = nchannels
        self.out_channels = 2 * nchannels

    def forward(self, x):
        x1 = torch.cat((x, -x), 1)
        x2 = self.scale(x1)
        y = self.relu(x2)
        return y

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nchannels': 4}]
