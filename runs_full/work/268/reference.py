import math
import torch
from torch import nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):

    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)

    def forward(self, hidden, encoder_outputs, mask=None):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = self.score(h, encoder_outputs)
        if mask is not None:
            attn_energies.data.masked_fill_(mask, -float('inf'))
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
