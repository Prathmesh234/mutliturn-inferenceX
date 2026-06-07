import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_gat(inp_ptr, W_ptr, a_ptr, adj_ptr, out_ptr, N, IN, OUT, TOTAL, alpha,
               BLOCK_N: tl.constexpr, BLOCK_IN: tl.constexpr, BLOCK_K: tl.constexpr,
               FINAL: tl.constexpr):
    hh = tl.program_id(0)
    i = tl.program_id(1)
    offs_n = tl.arange(0, BLOCK_N)
    offs_in = tl.arange(0, BLOCK_IN)
    offs_k = tl.arange(0, BLOCK_K)
    mask_n = offs_n < N
    mask_in = offs_in < IN
    mask_k = offs_k < OUT
    col = hh * OUT

    inp = tl.load(inp_ptr + offs_n[:, None] * IN + offs_in[None, :],
                  mask=mask_n[:, None] & mask_in[None, :], other=0.0)
    w = tl.load(W_ptr + offs_in[:, None] * TOTAL + (col + offs_k)[None, :],
                mask=mask_in[:, None] & mask_k[None, :], other=0.0)
    H = tl.dot(inp, w)  # (BLOCK_N, BLOCK_K) = input @ W_head

    a_base = a_ptr + hh * (2 * OUT)
    a1 = tl.load(a_base + offs_k, mask=mask_k, other=0.0)
    a2 = tl.load(a_base + OUT + offs_k, mask=mask_k, other=0.0)

    f = tl.sum(H * a1[None, :], axis=1)            # (BLOCK_N,)
    f_i = tl.sum(tl.where(offs_n == i, f, 0.0))
    g = tl.sum(H * a2[None, :], axis=1)            # (BLOCK_N,)
    e = f_i + g
    e = tl.where(e > 0, e, alpha * e)              # leakyrelu(slope=alpha)

    adj_i = tl.load(adj_ptr + i * N + offs_n, mask=mask_n, other=0.0)
    e = tl.where(adj_i > 0, e, -9e15)
    e = tl.where(mask_n, e, float('-inf'))

    m = tl.max(e)
    num = tl.exp(e - m)
    denom = tl.sum(num)
    att = num / denom

    h_prime = tl.sum(att[:, None] * H, axis=0)     # (BLOCK_K,)
    h_prime = tl.where(h_prime > 0, h_prime, tl.exp(h_prime) - 1.0)  # elu
    if FINAL:
        hp = tl.where(mask_k, h_prime, float('-inf'))
        mm = tl.max(hp)
        s = tl.sum(tl.exp(hp - mm))
        h_prime = (hp - mm) - tl.log(s)            # log_softmax over OUT
    tl.store(out_ptr + i * TOTAL + col + offs_k, h_prime, mask=mask_k)


def _pow16(x):
    return max(16, triton.next_power_of_2(x))


def _gat_multi(input, adj, W_all, a_all, in_features, out_features, nheads, alpha,
               final=False):
    N = input.shape[0]
    total = out_features * nheads
    out = torch.empty((N, total), device=input.device, dtype=torch.float32)
    _fused_gat[(nheads, N)](input, W_all, a_all, adj, out, N, in_features,
                            out_features, total, float(alpha),
                            BLOCK_N=_pow16(N), BLOCK_IN=_pow16(in_features),
                            BLOCK_K=_pow16(out_features), FINAL=final, num_warps=4)
    return out


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)


class PetarVGATNew(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(PetarVGATNew, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.nhid = nhid
        self.nfeat = nfeat
        self.alpha = alpha
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout,
            alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=
            dropout, alpha=alpha, concat=False)

    def _cache(self):
        c = getattr(self, '_pc', None)
        if c is None:
            c = (
                torch.cat([att.W for att in self.attentions], dim=1).contiguous(),
                torch.cat([att.a.view(-1) for att in self.attentions]).contiguous(),
                self.out_att.W.contiguous(),
                self.out_att.a.view(-1).contiguous(),
            )
            self._pc = c
        return c

    def forward(self, x, adj):
        x = x.contiguous()
        adj = adj.contiguous()
        W_all, a_all, oW, oa = self._cache()
        x = _gat_multi(x, adj, W_all, a_all, self.nfeat, self.nhid, self.nheads,
                       self.alpha)
        x = _gat_multi(x, adj, oW, oa, self.nhid * self.nheads,
                       self.out_att.out_features, 1, self.out_att.alpha, final=True)
        return x
