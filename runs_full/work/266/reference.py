import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResConv(nn.Module):

    def __init__(self, channels, dilation=1, activation='relu', padding=1,
        kernel_size=3, left_pad=0):
        super().__init__()
        in_channels = channels
        if activation == 'relu':
            self.activation = lambda *args, **kwargs: F.relu(*args, **
                kwargs, inplace=True)
        elif activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'glu':
            self.activation = F.glu
            in_channels = channels // 2
        self.left_pad = left_pad
        self.dilated_conv = nn.Conv1d(in_channels, channels, kernel_size=
            kernel_size, stride=1, padding=dilation * padding, dilation=
            dilation, bias=True)
        self.conv_1x1 = nn.Conv1d(in_channels, channels, kernel_size=1,
            bias=True)

    def forward(self, input):
        x = input
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))
        x = self.dilated_conv(x)
        x = self.activation(x)
        x = self.conv_1x1(x)
        return input + x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
