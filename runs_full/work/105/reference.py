import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, input_dim, output_class_num, **kwargs):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_dim, output_class_num)

    def forward(self, features):
        pooled = features.mean(dim=1)
        predicted = self.linear(pooled)
        return predicted


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_class_num': 4}]
