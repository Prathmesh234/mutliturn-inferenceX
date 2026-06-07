import torch
import numpy as np
import torch.nn as nn


class ExpNormalBasis(nn.Module):

    def __init__(self, n_rbf, cutoff, learnable_mu, learnable_beta):
        super().__init__()
        self.mu = torch.linspace(np.exp(-cutoff), 1, n_rbf)
        init_beta = (2 / n_rbf * (1 - np.exp(-cutoff))) ** -2
        self.beta = torch.ones_like(self.mu) * init_beta
        if learnable_mu:
            self.mu = nn.Parameter(self.mu)
        if learnable_beta:
            self.beta = nn.Parameter(self.beta)
        self.cutoff = cutoff

    def forward(self, dist):
        """
        Args:
            d (torch.Tensor): tensor of distances
        """
        shape_d = dist.unsqueeze(-1)
        mu = self.mu
        beta = self.beta
        arg = beta * (torch.exp(-shape_d) - mu) ** 2
        output = torch.exp(-arg)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_rbf': 4, 'cutoff': 4, 'learnable_mu': 4,
        'learnable_beta': 4}]
