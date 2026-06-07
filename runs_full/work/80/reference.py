import torch
import torch.nn as nn


class CasualConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        dilation=1, groups=1, bias=True):
        super(CasualConv1d, self).__init__()
        self.dilation = dilation
        padding = dilation * (kernel_size - 1)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)

    def forward(self, input):
        out = self.conv1d(input)
        return out[:, :, :-self.dilation]


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
