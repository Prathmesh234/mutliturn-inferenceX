import torch
import torch.utils.data
import torch.nn as nn


class Deep_Neural_Network(nn.Module):

    def __init__(self, D_in, fc1_size=40, fc2_size=20, fc3_size=40,
        fc4_size=20, fc5_size=40):
        """
        Neural Network model with 1 hidden layer.

        D_in: Dimension of input
        fc1_size, fc2_size, etc.: Dimensions of respective hidden layers
        """
        super(Deep_Neural_Network, self).__init__()
        self.fc1 = nn.Linear(D_in, fc1_size)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        nn.init.kaiming_normal_(self.fc3.weight)
        self.relu3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(fc3_size, fc4_size)
        nn.init.kaiming_normal_(self.fc4.weight)
        self.relu4 = nn.LeakyReLU()
        self.fc5 = nn.Linear(fc4_size, fc5_size)
        nn.init.kaiming_normal_(self.fc5.weight)
        self.relu5 = nn.LeakyReLU()
        self.fc_output = nn.Linear(fc5_size, 1)
        self.fc_output_activation = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        Forward function acceps a Tensor of input data and returns a tensor of output data.
        """
        out = self.dropout(self.relu1(self.fc1(x)))
        out = self.dropout(self.relu2(self.fc2(out)))
        out = self.dropout(self.relu3(self.fc3(out)))
        out = self.dropout(self.relu4(self.fc4(out)))
        out = self.dropout(self.relu5(self.fc5(out)))
        out = self.fc_output_activation(self.fc_output(out))
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'D_in': 4}]
