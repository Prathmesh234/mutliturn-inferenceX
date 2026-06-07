import torch
import torch.nn.functional as F
import torch.nn as nn


class BPNet(nn.Module):

    def __init__(self, input_dim, output_dim, level1, level2):
        super(BPNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, level1)
        self.fc2 = nn.Linear(level1, level2)
        self.fc3 = nn.Linear(level2, output_dim)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4, 'level1': 4, 'level2': 4}]
