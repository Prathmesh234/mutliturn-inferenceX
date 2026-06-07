import torch
import torch.nn as nn


class Downsample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=3,
            stride=2, padding=1)
        self.bn1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv3d(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
