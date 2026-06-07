import torch
from torch import nn
import torch.nn.functional as F


class SmallNN(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.l1 = nn.Linear(in_channels, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, out_channels)

    def forward(self, xb):
        a1 = F.relu(self.l1(xb))
        a2 = F.relu(self.l2(a1))
        return self.l3(a2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
