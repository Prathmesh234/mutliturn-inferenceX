import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Hardswish(nn.Module):

    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
