import torch
import torch.nn as nn
from torch.nn import functional as F


class ResizeConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor,
        mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
            stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4,
        'scale_factor': 1.0}]
