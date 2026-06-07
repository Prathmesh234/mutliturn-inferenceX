import torch
import torch.nn as nn


class InvDepth(nn.Module):

    def __init__(self, height, width, min_depth=0.5, max_depth=25.0):
        super(InvDepth, self).__init__()
        self._min_range = 1.0 / max_depth
        self._max_range = 1.0 / min_depth
        self.w = nn.Parameter(self._init_weights(height, width))

    def _init_weights(self, height, width):
        r1 = self._min_range
        r2 = self._min_range + (self._max_range - self._min_range) * 0.1
        w_init = (r1 - r2) * torch.rand(1, 1, height, width) + r2
        return w_init

    def forward(self):
        return self.w.clamp(min=self._min_range, max=self._max_range)


def get_inputs():
    return []


def get_init_inputs():
    return [[], {'height': 4, 'width': 4}]
