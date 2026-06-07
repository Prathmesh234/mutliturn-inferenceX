import torch
import torch.nn as nn
import torch.nn.functional as F


class PaddedMaxPool2d(nn.Module):
    """ Maxpool layer with a replicating padding.

    Args:
        kernel_size (int or tuple): Kernel size for maxpooling
        stride (int or tuple, optional): The stride of the window; Default ``kernel_size``
        padding (tuple, optional): (left, right, top, bottom) padding; Default **None**
        dilation (int or tuple, optional): A parameter that controls the stride of elements in the window
    """

    def __init__(self, kernel_size, stride=None, padding=(0, 0, 0, 0),
        dilation=1):
        super(PaddedMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation

    def __repr__(self):
        return (
            f'{self.__class__.__name__} (kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation})'
            )

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, self.padding, mode='replicate'), self.
            kernel_size, self.stride, 0, self.dilation)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
