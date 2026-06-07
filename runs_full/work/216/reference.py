import torch
import torch.nn.functional
import torch.nn as nn


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):

    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class ConvRelu2(nn.Module):

    def __init__(self, _in, _out):
        super(ConvRelu2, self).__init__()
        self.cr1 = ConvRelu(_in, _out)
        self.cr2 = ConvRelu(_out, _out)

    def forward(self, x):
        x = self.cr1(x)
        x = self.cr2(x)
        return x


class Coder(nn.Module):

    def __init__(self, in_size, out_size):
        super(Coder, self).__init__()
        self.conv = ConvRelu2(in_size, out_size)
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, x):
        y1 = self.conv(x)
        y2 = self.down(y1)
        return y2, y1


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4, 'out_size': 4}]
