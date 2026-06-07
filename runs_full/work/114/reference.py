import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch


class MaskedConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
        groups=1, bias=True, causal=True):
        if causal:
            padding = (kernel_size - 1) * dilation
        else:
            padding = (kernel_size - 1) * dilation // 2
        super(MaskedConv1d, self).__init__(in_channels, out_channels,
            kernel_size, stride=1, padding=padding, dilation=dilation,
            groups=groups, bias=bias)

    def forward(self, inputs):
        output = super(MaskedConv1d, self).forward(inputs)
        return output[:, :, :inputs.size(2)]


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
