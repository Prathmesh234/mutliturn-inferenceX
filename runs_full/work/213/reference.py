import torch
import torch.nn as nn


class conv2(nn.Module):

    def __init__(self, num_classes=2, in_channels=3, is_deconv=False,
        is_batchnorm=False, *args, **kwargs):
        super(conv2, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.final = nn.Conv2d(self.in_channels, num_classes, 1)

    def forward(self, inputs):
        return self.final(inputs)


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
