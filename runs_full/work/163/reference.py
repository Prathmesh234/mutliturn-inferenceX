import math
import torch
import numpy as np
from torch import nn


class MaybeBatchNorm2d(nn.Module):

    def __init__(self, n_ftr, affine, use_bn):
        super().__init__()
        self.bn = nn.BatchNorm2d(n_ftr, affine=affine)
        self.use_bn = use_bn

    def forward(self, x):
        if self.use_bn:
            x = self.bn(x)
        return x


class FakeRKHSConvNet(nn.Module):

    def __init__(self, n_input, n_output, use_bn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(n_input, n_output, kernel_size=1, stride=1,
            padding=0, bias=False)
        self.bn1 = MaybeBatchNorm2d(n_output, True, use_bn)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_output, n_output, kernel_size=1, stride=1,
            padding=0, bias=False)
        self.bn_out = MaybeBatchNorm2d(n_output, True, True)
        self.shortcut = nn.Conv2d(n_input, n_output, kernel_size=1, stride=
            1, padding=0, bias=True)
        if n_output >= n_input:
            eye_mask = np.zeros((n_output, n_input, 1, 1), dtype=np.bool)
            for i in range(n_input):
                eye_mask[i, i, 0, 0] = 1
            self.shortcut.weight.data.uniform_(-0.01, 0.01)
            self.shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.0)

    def init_weights(self, init_scale=1.0):
        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        self.conv1.weight.data.mul_(init_scale)
        nn.init.constant_(self.conv2.weight, 0.0)

    def forward(self, x):
        h_res = self.conv2(self.relu1(self.bn1(self.conv1(x))))
        h = self.bn_out(h_res + self.shortcut(x))
        return h


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_input': 4, 'n_output': 4}]
