import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.autograd


class Net(nn.Module):

    def __init__(self, STATE_NUM, ACTION_NUM):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features=STATE_NUM, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=ACTION_NUM)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.fc2(x)
        return action_value


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'STATE_NUM': 4, 'ACTION_NUM': 4}]
