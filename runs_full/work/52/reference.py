import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(self, hidden_size, attention_size=100, n_attention_heads=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.n_attention_heads = n_attention_heads
        self.W1 = nn.Linear(hidden_size, attention_size, bias=False)
        self.W2 = nn.Linear(attention_size, n_attention_heads, bias=False)

    def forward(self, hidden):
        hidden = hidden.transpose(0, 1)
        x = torch.tanh(self.W1(hidden))
        x = F.softmax(self.W2(x), dim=1)
        A = x.transpose(1, 2)
        M = A @ hidden
        return M, A


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
