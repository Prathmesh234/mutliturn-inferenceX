import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter


class ConstAttention(nn.Module):

    def __init__(self, **kwargs):
        super(ConstAttention, self).__init__()

    def forward(self, neighbor_vecs, self_vecs):
        return 1


class GatAttention(ConstAttention):

    def __init__(self, num_heads, out_channels):
        super(GatAttention, self).__init__()
        self.num_heads = num_heads
        self.out_channels = out_channels
        self.att_self_weight = Parameter(torch.Tensor(1, self.num_heads,
            self.out_channels))
        self.att_neighbor_weight = Parameter(torch.Tensor(1, self.num_heads,
            self.out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, neighbor_vecs, self_vecs):
        alpha = (self_vecs * self.att_self_weight).sum(dim=-1) + (neighbor_vecs
             * self.att_neighbor_weight).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        return alpha


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_heads': 4, 'out_channels': 4}]
