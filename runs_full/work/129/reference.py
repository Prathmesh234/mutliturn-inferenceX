import torch
import torch.nn as nn
import torch.nn.functional as F


class make_residual_dense_ver2(nn.Module):

    def __init__(self, nChannels, nChannels_, growthRate, kernel_size=3):
        super(make_residual_dense_ver2, self).__init__()
        if nChannels == nChannels_:
            self.conv = nn.Conv2d(nChannels_, growthRate, kernel_size=
                kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        else:
            self.conv = nn.Conv2d(nChannels_ + growthRate, growthRate,
                kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                bias=False)
        self.nChannels_ = nChannels_
        self.nChannels = nChannels
        self.growthrate = growthRate

    def forward(self, x):
        outoflayer = F.relu(self.conv(x))
        if x.shape[1] == self.nChannels:
            out = torch.cat((x, x + outoflayer), 1)
        else:
            out = torch.cat((x[:, :self.nChannels, :, :], x[:, self.
                nChannels:self.nChannels + self.growthrate, :, :] +
                outoflayer, x[:, self.nChannels + self.growthrate:, :, :]), 1)
        out = torch.cat((out, outoflayer), 1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nChannels': 4, 'nChannels_': 4, 'growthRate': 4}]
