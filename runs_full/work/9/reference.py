import torch
import torch.nn as nn


class Conv1d(nn.Module):

    def __init__(self, nf, nx, stdev=0.02):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.stdev = stdev
        self.w = nn.Parameter(torch.normal(size=[1, self.nx, self.nf], mean
            =0.0, std=self.stdev))
        self.b = nn.Parameter(torch.zeros([self.nf]))

    def forward(self, x: 'torch.Tensor'):
        shape = x.size()
        start, nx = shape[:-1], shape[-1]
        return torch.reshape(torch.matmul(torch.reshape(x, [-1, nx]), torch
            .reshape(self.w, [-1, self.nf])) + self.b, start + (self.nf,))


class Mlp(nn.Module):

    def __init__(self, input_dim, proj_dim):
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.conv_fc = Conv1d(self.proj_dim, self.input_dim)
        self.conv_proj = Conv1d(self.input_dim, self.proj_dim)

    def forward(self, x):
        h = nn.functional.gelu(self.conv_fc(x))
        return self.conv_proj(h)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'proj_dim': 4}]
