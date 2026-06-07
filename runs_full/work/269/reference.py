import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPool2dStaticSamePadding(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size
        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1
            ] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0
            ] - h + self.kernel_size[0]
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top
        x = F.pad(x, [left, right, top, bottom])
        x = self.pool(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
