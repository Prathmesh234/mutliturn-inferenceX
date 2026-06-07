import torch
import torch.nn as nn
import torch.nn.functional as F


class make_dense(nn.Module):

    def __init__(self, nChannels, nChannels_, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels_, growthRate, kernel_size=
            kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.nChannels = nChannels

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nChannels': 4, 'nChannels_': 4, 'growthRate': 4}]
