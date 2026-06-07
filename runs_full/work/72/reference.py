import math
import torch
from torch import nn
import torch.nn.functional as F


class MaxPool2dDynamicSamePadding(nn.MaxPool2d):
    """2D MaxPooling like TensorFlow's 'SAME' mode, with a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """

    def __init__(self, kernel_size, stride, padding=0, dilation=1,
        return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation,
            return_indices, ceil_mode)
        self.stride = [self.stride] * 2 if isinstance(self.stride, int
            ) else self.stride
        self.kernel_size = [self.kernel_size] * 2 if isinstance(self.
            kernel_size, int) else self.kernel_size
        self.dilation = [self.dilation] * 2 if isinstance(self.dilation, int
            ) else self.dilation

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] +
            1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] +
            1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h -
                pad_h // 2])
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding,
            self.dilation, self.ceil_mode, self.return_indices)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4, 'stride': 1}]
