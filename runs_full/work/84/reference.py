import torch
import torch.nn as nn


class ATRCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(ATRCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._W = nn.Parameter(torch.FloatTensor(input_size, hidden_size))
        self._W_b = nn.Parameter(torch.FloatTensor(hidden_size))
        self._U = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        self._U_b = nn.Parameter(torch.FloatTensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._W.data)
        nn.init.xavier_uniform_(self._U.data)
        nn.init.constant_(self._W_b.data, 0)
        nn.init.constant_(self._U_b.data, 0)

    def forward(self, x, h_):
        p = torch.mm(x, self._W) + self._W_b
        q = torch.mm(h_, self._U) + self._U_b
        i = (p + q).sigmoid()
        f = (p - q).sigmoid()
        h = (i * p + f * h_).tanh()
        return h


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
