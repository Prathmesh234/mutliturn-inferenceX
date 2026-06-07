import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.autograd


class duelingdqnNet(nn.Module):

    def __init__(self, STATE_NUM, ACTION_NUM):
        super(duelingdqnNet, self).__init__()
        self.ACTION_NUM = ACTION_NUM
        self.fc1_a = nn.Linear(in_features=STATE_NUM, out_features=512)
        self.fc1_v = nn.Linear(in_features=STATE_NUM, out_features=512)
        self.fc2_a = nn.Linear(in_features=512, out_features=ACTION_NUM)
        self.fc2_v = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        a = F.relu(self.fc1_a(x))
        v = F.relu(self.fc1_v(x))
        a = self.fc2_a(a)
        v = self.fc2_v(v).expand(x.size(0), self.ACTION_NUM)
        x = a + v - a.mean(1).unsqueeze(1).expand(x.size(0), self.ACTION_NUM)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'STATE_NUM': 4, 'ACTION_NUM': 4}]
