import torch
from torch.nn import functional as F


class SchedulerTestNet(torch.nn.Module):
    """adapted from: https://github.com/pytorch/pytorch/blob/master/test/test_optim.py."""

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
