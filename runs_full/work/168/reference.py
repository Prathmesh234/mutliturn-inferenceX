import torch
from torch import nn
from torch.distributions import Normal


class NormalProposal(nn.Module):

    def __init__(self, sigma):
        super(NormalProposal, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        return Normal(x, self.sigma).sample()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'sigma': 4}]
