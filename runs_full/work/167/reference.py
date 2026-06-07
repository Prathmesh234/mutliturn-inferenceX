import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.cuda.amp import autocast as autocast
import torch._C
import torch.serialization


class InputInjection(nn.Module):
    """Downsampling module for CGNet."""

    def __init__(self, num_downsampling):
        super(InputInjection, self).__init__()
        self.pool = nn.ModuleList()
        for i in range(num_downsampling):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_downsampling': 4}]
