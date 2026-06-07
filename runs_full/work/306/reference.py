import torch
import torch.nn as nn


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding='auto', dilation=1, bias=False, norm=nn.Identity(),
        activation=nn.ReLU()):
        super(Conv2d, self).__init__()
        if padding == 'auto':
            kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation
                 - 1)
            pad_total = kernel_size_effective - 1
            padding = pad_total // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            bias=bias)
        if activation is not None:
            self.bn = nn.Sequential(norm, activation)
        else:
            self.bn = norm

    def forward(self, x):
        return self.bn(self.conv(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
