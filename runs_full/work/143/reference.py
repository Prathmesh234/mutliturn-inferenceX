import torch
from torch import Tensor
from torch import nn


class SELoss(nn.MSELoss):

    def __init__(self):
        super().__init__(reduction='none')

    def forward(self, inputs: 'Tensor', target: 'Tensor') ->Tensor:
        return super().forward(inputs, target).sum(1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
