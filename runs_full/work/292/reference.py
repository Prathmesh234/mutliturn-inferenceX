import torch
import torch.nn


class DummyModel(torch.nn.Module):

    def __init__(self, block):
        super(DummyModel, self).__init__()
        self.block = block
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv(x)


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {'block': 4}]
