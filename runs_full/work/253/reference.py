import torch
import torch.nn as nn


class CausalConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1,
        **kwargs):
        super(CausalConv1d, self).__init__(in_channels, out_channels,
            kernel_size, padding=dilation * (kernel_size - 1), dilation=
            dilation, **kwargs)

    def forward(self, input):
        out = super(CausalConv1d, self).forward(input)
        return out[:, :, :-self.padding[0]]


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
