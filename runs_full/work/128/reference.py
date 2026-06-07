import torch
import torch.nn as nn


class Scale_and_shift(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.weight * x + self.bias


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
