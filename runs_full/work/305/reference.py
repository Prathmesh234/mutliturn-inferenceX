import torch
import numpy as np
import torch.nn as nn


class CosineEnvelope(nn.Module):

    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, d):
        output = 0.5 * (torch.cos(np.pi * d / self.cutoff) + 1)
        exclude = d >= self.cutoff
        output[exclude] = 0
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'cutoff': 4}]
