import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gcn_fused_kernel(
    x_ptr, adj_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
    M, F_, H, C,
    sx0, sx1, sa0, sa1,
    sw1_0, sw1_1, sw2_0, sw2_1, so0, so1,
    USE_ELU: tl.constexpr,
    BM: tl.constexpr, BF: tl.constexpr, BH: tl.constexpr, BC: tl.constexpr,
):
    offs_m = tl.arange(0, BM)
    offs_f = tl.arange(0, BF)
    offs_h = tl.arange(0, BH)
    offs_c = tl.arange(0, BC)

    # load x [M,F], adj [M,M]
    x = tl.load(x_ptr + offs_m[:, None] * sx0 + offs_f[None, :] * sx1,
                mask=(offs_m[:, None] < M) & (offs_f[None, :] < F_), other=0.0)
    adj = tl.load(adj_ptr + offs_m[:, None] * sa0 + offs_m[None, :] * sa1,
                  mask=(offs_m[:, None] < M) & (offs_m[None, :] < M), other=0.0)
    # W1 [H,F] -> need W1^T [F,H]
    w1 = tl.load(w1_ptr + offs_h[:, None] * sw1_0 + offs_f[None, :] * sw1_1,
                 mask=(offs_h[:, None] < H) & (offs_f[None, :] < F_), other=0.0)
    w1t = tl.trans(w1)  # [F,H]

    support1 = tl.dot(x, w1t)  # [M,H]
    b1 = tl.load(b1_ptr + offs_h, mask=offs_h < H, other=0.0)
    support1 += b1[None, :]

    h1 = tl.dot(adj, support1)  # [M,H]
    if USE_ELU:
        h1 = tl.where(h1 > 0, h1, tl.exp(h1) - 1.0)

    w2 = tl.load(w2_ptr + offs_c[:, None] * sw2_0 + offs_h[None, :] * sw2_1,
                 mask=(offs_c[:, None] < C) & (offs_h[None, :] < H), other=0.0)
    w2t = tl.trans(w2)  # [H,C]
    support2 = tl.dot(h1, w2t)  # [M,C]
    b2 = tl.load(b2_ptr + offs_c, mask=offs_c < C, other=0.0)
    support2 += b2[None, :]

    out = tl.dot(adj, support2)  # [M,C]
    tl.store(out_ptr + offs_m[:, None] * so0 + offs_c[None, :] * so1, out,
             mask=(offs_m[:, None] < M) & (offs_c[None, :] < C))


def _pow2(n):
    p = 16
    while p < n:
        p *= 2
    return p


@triton.jit
def _mm_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr, M, N, K,
    sam, sak, sbk, sbn, scm, scn,
    HAS_BIAS: tl.constexpr, APPLY_ELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * sak
        b_ptrs += BLOCK_K * sbk
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
    if APPLY_ELU:
        acc = tl.where(acc > 0, acc, tl.exp(acc) - 1.0)
    tl.store(c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _mm(a, b, bias=None, apply_elu=False):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    BLOCK_M = _pow2(M); BLOCK_N = _pow2(N); BLOCK_K = _pow2(K)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _mm_kernel[grid](a, b, c, bias if bias is not None else a, M, N, K,
                     a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                     c.stride(0), c.stride(1),
                     HAS_BIAS=bias is not None, APPLY_ELU=apply_elu,
                     BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                     num_warps=4, num_stages=2)
    return c


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1.0 / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, apply_elu=False):
        support = _mm(input, self.W.weight.t(), bias=self.W.bias)
        return _mm(adj, support, apply_elu=apply_elu)


class GCNNew(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCNNew, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.elu = torch.nn.ELU(inplace=True)

    def forward(self, x, adj, use_relu=True):
        M, Fd = x.shape
        H = self.gc1.out_features
        C = self.gc2.out_features
        no_dropout = (not self.training) or self.dropout == 0
        if no_dropout and max(M, Fd, H, C) <= 64:
            out = torch.empty((M, C), device=x.device, dtype=torch.float32)
            BM = _pow2(M); BF = _pow2(Fd); BH = _pow2(H); BC = _pow2(C)
            w1 = self.gc1.W.weight; b1 = self.gc1.W.bias
            w2 = self.gc2.W.weight; b2 = self.gc2.W.bias
            _gcn_fused_kernel[(1,)](
                x, adj, w1, b1, w2, b2, out, M, Fd, H, C,
                x.stride(0), x.stride(1), adj.stride(0), adj.stride(1),
                w1.stride(0), w1.stride(1), w2.stride(0), w2.stride(1),
                out.stride(0), out.stride(1),
                USE_ELU=use_relu, BM=BM, BF=BF, BH=BH, BC=BC,
                num_warps=1, num_stages=1)
            return out
        x = self.gc1(x, adj, apply_elu=use_relu)
        x = F.dropout(x, self.dropout, training=self.training)
        return self.gc2(x, adj)
