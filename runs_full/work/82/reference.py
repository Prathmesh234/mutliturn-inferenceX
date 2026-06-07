import torch
from torch import nn
import torch.onnx


class Block(nn.Module):

    def __init__(self, in_channels, num_filters, kernel_size, pool_size):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size
            )
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'num_filters': 4, 'kernel_size': 4,
        'pool_size': 4}]
