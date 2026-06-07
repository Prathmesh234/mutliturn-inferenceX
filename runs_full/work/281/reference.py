import torch
import torch.utils.data
import torch.utils.data.dataloader


class MomentumNetSide(torch.nn.Module):

    def __init__(self, beta: 'float'):
        super(MomentumNetSide, self).__init__()
        self.beta = beta

    def forward(self, inp: 'torch.Tensor'):
        return inp * self.beta


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'beta': 4}]
