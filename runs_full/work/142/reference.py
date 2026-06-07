import torch
import torch.nn.functional as F
import torch.nn as nn


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        hidden_units = 512
        self.fc1 = nn.Linear(state_size, hidden_units)
        self.do1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.do2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.do3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(hidden_units, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.do1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.do2(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.do3(x)
        x = self.fc4(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4, 'seed': 4}]
