import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch


class LayerNorm1d(nn.Module):

    def __init__(self, num_features, eps=1e-06, affine=True):
        super(LayerNorm1d, self).__init__()
        self.eps = eps
        self.num_features = num_features
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.fill_(0.0)

    def forward(self, inputs):
        b, t, _ = list(inputs.size())
        mean = inputs.mean(2).view(b, t, 1).expand_as(inputs)
        input_centered = inputs - mean
        std = input_centered.pow(2).mean(2).add(self.eps).sqrt()
        output = input_centered / std.view(b, t, 1).expand_as(inputs)
        if self.affine:
            w = self.weight.view(1, 1, -1).expand_as(output)
            b = self.bias.view(1, 1, -1).expand_as(output)
            output = output * w + b
        return output


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
