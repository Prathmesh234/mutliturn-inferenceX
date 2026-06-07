import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim, 1, bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M)
        alpha = F.softmax(scale, dim=0).permute(1, 2, 0)
        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]
        return attn_pool, alpha


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
