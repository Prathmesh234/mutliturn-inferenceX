import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F


class NodeAdaptiveEncoder(nn.Module):

    def __init__(self, num_features, dropout=0.5):
        super(NodeAdaptiveEncoder, self).__init__()
        self.fc = nn.Parameter(torch.zeros(size=(num_features, 1)))
        nn.init.xavier_normal_(self.fc.data, gain=1.414)
        self.bf = nn.Parameter(torch.zeros(size=(1,)))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        h = torch.mm(x, self.fc) + self.bf
        h = F.sigmoid(h)
        h = self.dropout(h)
        return torch.where(x < 0, torch.zeros_like(x), x) + h * torch.where(
            x > 0, torch.zeros_like(x), x)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
