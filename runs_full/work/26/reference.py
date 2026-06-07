import torch
import torch.nn as nn


class MeanStd(nn.Module):

    def __init__(self):
        super(MeanStd, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        mean_x = torch.mean(x, dim=2)
        var_x = torch.mean(x ** 2, dim=2) - mean_x * mean_x
        return torch.cat([mean_x, var_x], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
