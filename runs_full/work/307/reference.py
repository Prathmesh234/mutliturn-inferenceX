import math
import torch
from torch import Tensor
import torch.nn as nn


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, num_heads: 'int', size: 'int', dropout: 'float'=0.1):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super(MultiHeadedAttention, self).__init__()
        assert size % num_heads == 0
        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads
        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)
        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: 'Tensor', v: 'Tensor', q: 'Tensor', mask: 'Tensor'
        =None):
        """
        Computes multi-headed attention.

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, 1, M]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q / math.sqrt(self.head_size)
        scores = torch.matmul(q, k.transpose(2, 3))
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        attention = self.softmax(scores)
        attention = self.dropout(attention)
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
            num_heads * self.head_size)
        output = self.output_layer(context)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_heads': 4, 'size': 4}]
