import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class SDPAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, dropout=0, causal=False):
        super(SDPAttention, self).__init__()
        self.causal = causal
        self.dropout = nn.Dropout(dropout)
        self.mask_q = None
        self.mask_k = None

    def set_mask_q(self, masked_tq):
        self.mask_q = masked_tq

    def set_mask_k(self, masked_tk):
        self.mask_k = masked_tk

    def forward(self, q, k, v):
        b_q, t_q, dim_q = list(q.size())
        b_k, t_k, dim_k = list(k.size())
        b_v, t_v, _dim_v = list(v.size())
        assert b_q == b_k and b_k == b_v
        assert dim_q == dim_k
        assert t_k == t_v
        b = b_q
        qk = torch.bmm(q, k.transpose(1, 2))
        qk.div_(dim_k ** 0.5)
        mask = None
        if self.causal:
            causal_mask = q.data.new(t_q, t_k).byte().fill_(1).triu_(1)
            mask = Variable(causal_mask.unsqueeze(0).expand(b, t_q, t_k),
                requires_grad=False)
        if self.mask_k is not None:
            mask_k = self.mask_k.unsqueeze(1).expand(b, t_q, t_k)
            mask = mask_k if mask is None else mask | mask_k
        if self.mask_q is not None:
            mask_q = self.mask_q.unsqueeze(2).expand(b, t_q, t_k)
            mask = mask_q if mask is None else mask | mask_q
        if mask is not None:
            qk.masked_fill_(mask, -1000000000.0)
        sm_qk = F.softmax(qk, dim=2)
        sm_qk = self.dropout(sm_qk)
        return torch.bmm(sm_qk, v), sm_qk


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {}]
