import torch
import torch.nn as nn


class Gate(torch.nn.Module):

    def __init__(self, out_planes):
        super(Gate, self).__init__()
        self.gate = nn.Parameter(torch.ones(1, out_planes, 1, 1),
            requires_grad=False)

    def forward(self, x):
        return self.gate * x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'out_planes': 4}]
