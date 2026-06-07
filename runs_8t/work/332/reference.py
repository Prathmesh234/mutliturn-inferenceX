import torch
import torch.nn as nn


class HLoss(nn.Module):

    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = torch.exp(x) * x
        b = -1.0 * b.sum(dim=1)
        return b


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
