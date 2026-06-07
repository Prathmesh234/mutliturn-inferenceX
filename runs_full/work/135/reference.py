import torch
from torch import nn
from torch.nn import functional as F


class Policy_Net(nn.Module):

    def __init__(self, observation_dim, action_dim):
        super(Policy_Net, self).__init__()
        self.fc1 = nn.Linear(observation_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, observation):
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'observation_dim': 4, 'action_dim': 4}]
