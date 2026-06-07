import torch
import torch.nn


class MultiplicationInverse(torch.nn.Module):

    def __init__(self, factor=2):
        super(MultiplicationInverse, self).__init__()
        self.factor = torch.nn.Parameter(torch.ones(1) * factor)

    def forward(self, x):
        return x * self.factor

    def inverse(self, y):
        return y / self.factor


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
