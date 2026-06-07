import torch
import torch.nn as nn


def mean_channels(F):
    assert F.dim() == 4
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert F.dim() == 4
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True
        ) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def activation(act_type, inplace=False, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.
            format(act_type))
    return layer


class LCCALayer(nn.Module):

    def __init__(self, channel):
        super(LCCALayer, self).__init__()
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c3 = nn.Conv2d(channel, channel // 4, kernel_size=3, padding=(
            3 - 1) // 2, bias=False)
        self.c32 = nn.Conv2d(channel // 4, channel, kernel_size=3, padding=
            (3 - 1) // 2, bias=False)
        self.act = activation('relu')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.c32(self.c3(y))
        return self.sigmoid(y)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channel': 4}]
