import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data


class SimpleMLP(nn.Module):

    def __init__(self, n_inputs, n_outputs, dropout_probability):
        super(SimpleMLP, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.dropout_probability = dropout_probability
        self.l1 = nn.Linear(self.n_inputs, self.n_inputs * 2)
        self.l2 = nn.Linear(self.n_inputs * 2, self.n_inputs * 4)
        self.l3 = nn.Linear(self.n_inputs * 4, self.n_inputs * 8)
        self.l4 = nn.Linear(self.n_inputs * 8, self.n_outputs)
        self.dropout = nn.Dropout(self.dropout_probability)

    def forward(self, X):
        X = F.relu(self.l1(X))
        X = self.dropout(X)
        X = F.relu(self.l2(X))
        X = self.dropout(X)
        X = F.relu(self.l3(X))
        X = self.dropout(X)
        X = self.l4(X)
        return X


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_inputs': 4, 'n_outputs': 4, 'dropout_probability': 0.5}]
