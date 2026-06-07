import torch
import torch.nn as nn


class ConvRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(ConvRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, bias=bias)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
