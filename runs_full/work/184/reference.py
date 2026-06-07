import torch
import torch.utils.data
import torch.nn as nn


class Neural_Net(nn.Module):

    def __init__(self, D_in):
        """
        Neural Network model with 1 hidden layer.

        D_in: Dimension of input
        """
        super(Neural_Net, self).__init__()
        self.fc1 = nn.Linear(D_in, 100)
        self.relu1 = nn.Sigmoid()
        self.fc2 = nn.Linear(100, 50)
        self.relu2 = nn.Sigmoid()
        self.fc3 = nn.Linear(50, 20)
        self.relu3 = nn.ReLU()
        self.fc_output = nn.Linear(20, 1)
        self.fc_output_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc_output_activation(self.fc_output(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'D_in': 4}]
