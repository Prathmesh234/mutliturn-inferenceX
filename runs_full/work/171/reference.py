import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.cuda.amp import autocast as autocast
import torch._C
import torch.serialization


class ExampleBackbone(nn.Module):

    def __init__(self):
        super(ExampleBackbone, self).__init__()
        self.conv = nn.Conv2d(3, 3, 3)

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        return [self.conv(x)]


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
