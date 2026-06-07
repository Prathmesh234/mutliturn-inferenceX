import torch
import torch.nn as nn
import torch.nn.functional as F


def seq_dropout(x, p=0, training=False):
    """
    x: batch * len * input_size
    """
    if training is False or p == 0:
        return x
    dropout_mask = 1.0 / (1 - p) * torch.bernoulli((1 - p) * (x.new_zeros(x
        .size(0), x.size(2)) + 1))
    return dropout_mask.unsqueeze(1).expand_as(x) * x


def dropout(x, p=0, training=False):
    """
    x: (batch * len * input_size) or (any other shape)
    """
    if do_seq_dropout and len(x.size()) == 3:
        return seq_dropout(x, p=p, training=training)
    else:
        return F.dropout(x, p=p, training=training)


class MultiHeadAttn(nn.Module):

    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
        pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.scale = 1 / d_head ** 0.5
        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h
        if self.pre_lnorm:
            c = self.layer_norm(c)
        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)
        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float
                    ('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf')
                    )
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.
            size(1), self.n_head * self.d_head)
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        if self.pre_lnorm:
            output = h + attn_out
        else:
            output = self.layer_norm(h + attn_out)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'n_head': 4, 'd_model': 4, 'd_head': 4, 'dropout': 0.5}]
