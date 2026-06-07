import torch
import numpy as np
import torch.nn as nn


class PainnRadialBasis(nn.Module):

    def __init__(self, n_rbf, cutoff, learnable_k):
        super().__init__()
        self.n = torch.arange(1, n_rbf + 1).float()
        if learnable_k:
            self.n = nn.Parameter(self.n)
        self.cutoff = cutoff

    def forward(self, dist):
        """
        Args:
            d (torch.Tensor): tensor of distances
        """
        shape_d = dist.unsqueeze(-1)
        n = self.n
        coef = n * np.pi / self.cutoff
        device = shape_d.device
        denom = torch.where(shape_d == 0, torch.tensor(1.0, device=device),
            shape_d)
        num = torch.where(shape_d == 0, coef, torch.sin(coef * shape_d))
        output = torch.where(shape_d >= self.cutoff, torch.tensor(0.0,
            device=device), num / denom)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_rbf': 4, 'cutoff': 4, 'learnable_k': 4}]
