import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils


class Bc(nn.Module):

    def __init__(self, nc):
        super(Bc, self).__init__()
        self.nn = nn.Linear(nc, 1)

    def forward(self, input):
        return torch.sigmoid(self.nn(input))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nc': 4}]
