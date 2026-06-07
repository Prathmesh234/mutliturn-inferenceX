import torch
from torch.optim.lr_scheduler import *
import torch.nn.functional as F
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.utils.model_zoo


class MaxPoolStride1(nn.Module):

    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0, 1, 0, 1), mode='replicate'), 2, stride=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
