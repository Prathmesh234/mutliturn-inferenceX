import torch
import torch.nn


class SplitChannels(torch.nn.Module):

    def __init__(self, split_location):
        self.split_location = split_location
        super(SplitChannels, self).__init__()

    def forward(self, x):
        return x[:, :self.split_location, :].clone(), x[:, self.
            split_location:, :].clone()

    def inverse(self, x, y):
        return torch.cat([x, y], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'split_location': 4}]
