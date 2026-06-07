import torch
import torch.nn as nn
import torch.nn.functional as F


class make_dense_LReLU(nn.Module):

    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense_LReLU, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=
            kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nChannels': 4, 'growthRate': 4}]
