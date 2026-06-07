import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import *
from torch.optim import *


class TLU(nn.Module):

    def __init__(self, num_features):
        """max(y, tau) = max(y - tau, 0) + tau = ReLU(y - tau) + tau"""
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = Parameter(torch.Tensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)

    def extra_repr(self):
        return 'num_features={num_features}'.format(**self.__dict__)

    def forward(self, x):
        return torch.max(x, self.tau.view(1, self.num_features, 1, 1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
