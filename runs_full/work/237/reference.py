import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.autograd


class MuNet(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        return mu


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
