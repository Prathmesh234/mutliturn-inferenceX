import torch
import torch.nn as nn
import torch.utils.checkpoint


class SkipConnection(nn.Module):
    """Linearize gradients, to make learning easier."""

    def __init__(self, *fn):
        super().__init__()
        self.fn = nn.Sequential(*fn)

    def forward(self, x):
        y = self.fn(x)
        return y if x.shape[-1] < y.shape[-1
            ] else x + y if x.shape == y.shape else x[..., :y.shape[-1]] + y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
