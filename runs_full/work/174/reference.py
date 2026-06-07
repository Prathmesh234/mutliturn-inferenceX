import torch
import torch.nn as nn


class Gblock(nn.Module):

    def __init__(self, in_channels, out_channels, groups):
        super(Gblock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            padding=1, groups=groups)
        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
            padding=0)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu(x)
        x = self.conv1(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'groups': 1}]
