import torch
import torch.nn.functional as F
import torch.nn as nn


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(2, 4, kernel_size=6, stride=2, bias=False)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
        self.size = 9 * 9 * 16
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))
        return self.sig(self.fc2(x))


def get_inputs():
    return [torch.rand([4, 2, 81, 81])]


def get_init_inputs():
    return [[], {}]
