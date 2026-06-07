import torch
import torch.nn as nn


class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.trilinear = nn.Upsample(scale_factor=scale_factor)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.trilinear(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
