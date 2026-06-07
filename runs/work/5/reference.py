import torch
import torch.nn as nn


class Norm(nn.Module):

    def __init__(self, n_state, axis=-1, epsilon=1e-05):
        super().__init__()
        self.n_state = n_state
        self.g = nn.Parameter(torch.ones([self.n_state]))
        self.b = nn.Parameter(torch.zeros([self.n_state]))
        self.axis = axis
        self.epsilon = epsilon

    def forward(self, x):
        u = torch.mean(x, dim=self.axis, keepdim=True)
        s = torch.mean(torch.square(x - u), dim=self.axis, keepdim=True)
        x = (x - u) * torch.rsqrt(s + self.epsilon)
        x = x * self.g + self.b
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_state': 4}]
