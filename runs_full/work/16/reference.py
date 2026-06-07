import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnectedNet(nn.Module):
    """https://github.com/VainF/Torch-Pruning/issues/21"""

    def __init__(self, input_size, num_classes, HIDDEN_UNITS):
        super().__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_UNITS)
        self.fc2 = nn.Linear(HIDDEN_UNITS, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        y_hat = self.fc2(x)
        return y_hat


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'num_classes': 4, 'HIDDEN_UNITS': 4}]
