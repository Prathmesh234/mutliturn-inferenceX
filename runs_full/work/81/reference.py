import torch
import torch.nn as nn


class CasualConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        dilation=1, groups=1, bias=True):
        super(CasualConv1d, self).__init__()
        self.dilation = dilation
        padding = dilation * (kernel_size - 1)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)

    def forward(self, input):
        out = self.conv1d(input)
        return out[:, :, :-self.dilation]


class DenseBlock(nn.Module):

    def __init__(self, in_channels, dilation, filters, kernel_size=2):
        super(DenseBlock, self).__init__()
        self.casualconv1 = CasualConv1d(in_channels, filters, kernel_size,
            dilation=dilation)
        self.casualconv2 = CasualConv1d(in_channels, filters, kernel_size,
            dilation=dilation)

    def forward(self, input):
        xf = self.casualconv1(input)
        xg = self.casualconv2(input)
        activations = torch.tanh(xf) * torch.sigmoid(xg)
        return torch.cat((input, activations), dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'dilation': 1, 'filters': 4}]
