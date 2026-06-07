import math
import torch
from torch import nn


class Attention(nn.Module):
    """A generic attention module for a decoder in seq2seq"""

    def __init__(self, dim, use_tanh=False, C=10):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C
        self.tanh = nn.Tanh()
        self.v = nn.Parameter(torch.FloatTensor(dim))
        self.v.data.uniform_(-(1.0 / math.sqrt(dim)), 1.0 / math.sqrt(dim))

    def forward(self, query, ref):
        """
        Args: 
            query: is the hidden state of the decoder at the current
                time step. batch x dim
            ref: the set of hidden states from the encoder. 
                sourceL x batch x hidden_dim
        """
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)
        e = self.project_ref(ref)
        expanded_q = q.repeat(1, 1, e.size(2))
        v_view = self.v.unsqueeze(0).expand(expanded_q.size(0), len(self.v)
            ).unsqueeze(1)
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u
        return e, logits


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
