import torch
import torch.nn.functional
import torch.nn as nn
from torch.nn import functional as F


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


class Decoder(nn.Module):

    def __init__(self, in_size, out_size):
        super(Decoder, self).__init__()
        self.conv = ConvRelu2(in_size, out_size)
        self.up = F.interpolate

    def forward(self, x1, x2):
        x2 = self.up(x2, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x1, x2], 1))


class AttentionNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1):
        super(AttentionNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        filters = [64, 128, 256]
        self.down1 = Coder(in_channels, filters[0])
        self.down2 = Coder(filters[0], filters[1])
        self.center = ConvRelu2(filters[1], filters[2])
        self.up2 = Decoder(filters[2] + filters[1], filters[1])
        self.up1 = Decoder(filters[1] + filters[0], filters[0])
        self.final = nn.Conv2d(filters[0], out_channels, 1)

    def forward(self, x):
        x, befdown1 = self.down1(x)
        x, befdown2 = self.down2(x)
        x = self.center(x)
        x = self.up2(befdown2, x)
        x = self.up1(befdown1, x)
        x = self.final(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
