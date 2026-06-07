import torch
import torch.nn as nn


class LRNCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LRNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._W = nn.Parameter(torch.FloatTensor(input_size, hidden_size * 3))
        self._W_b = nn.Parameter(torch.FloatTensor(hidden_size * 3))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._W.data)
        nn.init.constant_(self._W_b.data, 0)

    def forward(self, x, h_):
        p, q, r = (torch.mm(x, self._W) + self._W_b).split(self.hidden_size, -1
            )
        i = (p + h_).sigmoid()
        f = (q - h_).sigmoid()
        h = (i * r + f * h_).tanh()
        return h


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
