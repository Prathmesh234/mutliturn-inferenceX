import torch
import torch.nn as nn


class Conv2dTime(nn.Conv2d):
    """
    Implements time dependent 2d convolutions, by appending the time variable as
    an extra channel.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super(Conv2dTime, self).__init__(in_channels + 1, *args, **kwargs)

    def forward(self, t, x):
        t_img = torch.ones_like(x[:, :1, :, :]) * t
        t_and_x = torch.cat([t_img, x], 1)
        return super(Conv2dTime, self).forward(t_and_x)


def get_inputs():
    return [torch.rand([4, 1, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
