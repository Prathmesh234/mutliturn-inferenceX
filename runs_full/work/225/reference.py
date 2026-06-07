import torch
from torch import nn
from torch.optim.lr_scheduler import *
from torch.optim import *


class GlobalMaxPool(nn.AdaptiveMaxPool2d):

    def __init__(self, output_size=1, *args, **kwargs):
        super().__init__(output_size)


class FastGlobalAvgPool(nn.Module):

    def __init__(self, flatten=False, *args, **kwargs):
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0),
                x.size(1), 1, 1)


class AdaptiveAvgMaxPool(nn.Module):

    def __init__(self, output_size=1, *args, **kwargs):
        super().__init__()
        self.gap = FastGlobalAvgPool()
        self.gmp = GlobalMaxPool(output_size)

    def forward(self, x):
        avg_feat = self.gap(x)
        max_feat = self.gmp(x)
        feat = avg_feat + max_feat
        return feat


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
