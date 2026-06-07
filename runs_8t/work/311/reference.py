import math
import torch
import numpy as np
from torch import nn


class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, input_dim, embed_dim, val_dim=None, key_dim
        =None):
        super(MultiHeadAttention, self).__init__()
        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.norm_factor = 1 / math.sqrt(key_dim)
        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, queries, data=None, mask=None):
        """

        :param queries: queries (batch_size, n_query, input_dim)
        :param data: data (batch_size, task_size, input_dim)
        :param mask: mask (batch_size, n_query, task_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if data is None:
            data = queries
        batch_size, task_size, input_dim = data.size()
        n_query = queries.size(1)
        assert queries.size(0) == batch_size
        assert queries.size(2) == input_dim
        assert input_dim == self.input_dim, 'Wrong embedding dimension of input'
        queries_flat = data.contiguous().view(-1, input_dim)
        data_flat = queries.contiguous().view(-1, input_dim)
        shp = self.n_heads, batch_size, task_size, -1
        shp_q = self.n_heads, batch_size, n_query, -1
        Q = torch.matmul(data_flat, self.W_query).view(shp_q)
        K = torch.matmul(queries_flat, self.W_key).view(shp)
        V = torch.matmul(queries_flat, self.W_val).view(shp)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, task_size).expand_as(
                compatibility)
            compatibility[mask] = -np.inf
        attn = torch.softmax(compatibility, dim=-1)
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc
        heads = torch.matmul(attn, V)
        out = torch.mm(heads.permute(1, 2, 0, 3).contiguous().view(-1, self
            .n_heads * self.val_dim), self.W_out.view(-1, self.embed_dim)
            ).view(batch_size, n_query, self.embed_dim)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'n_heads': 4, 'input_dim': 4, 'embed_dim': 4}]
