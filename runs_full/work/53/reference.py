import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, heads, n_state):
        super().__init__()
        assert n_state % heads == 0
        self.heads = heads
        self.n_state = n_state
        self.depth = self.n_state // self.heads

    def split_heads(self, x: 'torch.Tensor', batch: 'int', seq_len: 'int'):
        x = x.reshape((batch, seq_len, self.heads, self.depth))
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x: 'torch.Tensor', batch: 'int', seq_len: 'int'):
        x = x.permute(0, 2, 1, 3)
        return x.reshape((batch, seq_len, self.n_state))


class Conv1d(nn.Module):

    def __init__(self, nf, nx, stdev=0.02):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.stdev = stdev
        self.w = nn.Parameter(torch.normal(size=[1, self.nx, self.nf], mean
            =0.0, std=self.stdev))
        self.b = nn.Parameter(torch.zeros([self.nf]))

    def forward(self, x: 'torch.Tensor'):
        shape = x.size()
        start, nx = shape[:-1], shape[-1]
        return torch.reshape(torch.matmul(torch.reshape(x, [-1, nx]), torch
            .reshape(self.w, [-1, self.nf])) + self.b, start + (self.nf,))


class RelativeAttention(Attention):

    def __init__(self, heads, n_state, max_sequence):
        super().__init__(heads, n_state)
        self.max_sequence = max_sequence
        self.c_attn = Conv1d(self.n_state * 3, self.n_state)
        self.c_proj = Conv1d(self.n_state, self.n_state)
        self.E = nn.Parameter(torch.Tensor(self.heads, self.max_sequence, 
            n_state // heads))
        nn.init.xavier_normal_(self.E)

    def relative_attn(self, q: 'torch.Tensor', E: 'torch.Tensor', batch:
        'int', seq_len: 'int'):
        q_ = q.permute(1, 0, 2, 3)
        q_ = q_.reshape(self.heads, batch * seq_len, self.depth)
        E = E[:, self.max_sequence - seq_len:]
        rel = q_ @ E.transpose(-1, -2)
        rel = rel.reshape(self.heads, batch, seq_len, seq_len)
        rel = torch.nn.functional.pad(rel, (1, 0), 'constant', 0)
        rel = rel.reshape(self.heads, batch, seq_len + 1, seq_len)
        rel = rel[:, :, 1:]
        rel = rel.permute(1, 0, 2, 3)
        return rel

    def multihead_attn(self, q: 'torch.Tensor', k: 'torch.Tensor', v:
        'torch.Tensor', batch, seq_len, mask=None):
        w = q @ k.transpose(-1, -2)
        w = w + self.relative_attn(q, self.E, batch, seq_len)
        w = w * (1 / self.depth ** (1 / 2))
        if mask is not None:
            w += mask
        w = w.softmax(-1)
        a = w @ v
        return a

    def forward(self, x: 'torch.Tensor', mask=None):
        batch, seq_len, _ = x.size()
        c = self.c_attn(x)
        q, k, v = torch.split(c, self.n_state, dim=2)
        q = self.split_heads(q, batch, seq_len)
        k = self.split_heads(k, batch, seq_len)
        v = self.split_heads(v, batch, seq_len)
        a = self.multihead_attn(q, k, v, batch, seq_len, mask)
        a = self.combine_heads(a, batch, seq_len)
        a = self.c_proj(a)
        return a


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'heads': 4, 'n_state': 4, 'max_sequence': 4}]
