import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init


class ScalarScaleBias(nn.Module):

    def __init__(self, scale=True, scale_init=1.0, bias=True, bias_init=0.0
        ) ->None:
        super(ScalarScaleBias, self).__init__()
        if scale:
            self.weight = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('weight', None)
        if bias:
            self.bias = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('bias', None)
        self.weight_init = scale_init
        self.bias_init = bias_init
        self.reset_parameters()

    def reset_parameters(self) ->None:
        if self.weight is not None:
            init.constant_(self.weight, self.weight_init)
        if self.bias is not None:
            init.constant_(self.bias, self.bias_init)

    def forward(self, x):
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
