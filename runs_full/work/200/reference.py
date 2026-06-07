from torch.nn import Module
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional
from torch.nn import init
from torch.nn.modules import Module
import torch.utils.data


class NAC(Module):

    def __init__(self, n_in, n_out):
        super().__init__()
        self.W_hat = Parameter(torch.Tensor(n_out, n_in))
        self.M_hat = Parameter(torch.Tensor(n_out, n_in))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.W_hat)
        init.kaiming_uniform_(self.M_hat)

    def forward(self, input):
        weights = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        return functional.linear(input, weights)


class NALU(Module):

    def __init__(self, n_in, n_out):
        super().__init__()
        self.NAC = NAC(n_in, n_out)
        self.G = Parameter(torch.Tensor(1, n_in))
        self.eps = 1e-06
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.G)

    def forward(self, input):
        g = torch.sigmoid(functional.linear(input, self.G))
        y1 = g * self.NAC(input)
        y2 = (1 - g) * torch.exp(self.NAC(torch.log(torch.abs(input) + self
            .eps)))
        return y1 + y2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_in': 4, 'n_out': 4}]
