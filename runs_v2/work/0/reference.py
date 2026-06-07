import torch
import torch.nn as nn


class SumAggregator(nn.Module):

    def __init__(self):
        super(SumAggregator, self).__init__()

    def forward(self, neighbor):
        return torch.sum(neighbor, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
