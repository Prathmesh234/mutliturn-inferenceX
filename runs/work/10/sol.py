import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mm_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr, ELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_mask = offs_k[None, :] < K - k
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=((offs_k[:, None] < K - k) & (offs_n[None, :] < N)), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    if ELU:
        acc = tl.where(acc > 0, acc, tl.exp(acc) - 1.0)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _mm(a, b, bias=None, elu=False):
    M, K = a.shape
    K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    BLOCK_M = 64 if M > 32 else triton.next_power_of_2(M)
    BLOCK_N = 64 if N > 32 else triton.next_power_of_2(N)
    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_K = max(16, min(BLOCK_K, 64))
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _mm_kernel[grid](
        a, b, bias if bias is not None else a, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        HAS_BIAS=bias is not None, ELU=elu,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return c


@triton.jit
def _fused_kernel(
    x_ptr, adj_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
    M, NF, NH, NC,
    sx0, sx1, sa0, sa1,
    sw1_0, sw1_1, sw2_0, sw2_1,
    so0, so1,
    USE_RELU: tl.constexpr,
    BM: tl.constexpr, BF: tl.constexpr, BH: tl.constexpr, BC: tl.constexpr,
):
    rm = tl.arange(0, BM)
    rf = tl.arange(0, BF)
    rh = tl.arange(0, BH)
    rc = tl.arange(0, BC)

    # load x [BM, BF]
    x = tl.load(x_ptr + rm[:, None] * sx0 + rf[None, :] * sx1,
                mask=(rm[:, None] < M) & (rf[None, :] < NF), other=0.0)
    # w1 [NH, NF] -> need w1^T [BF, BH]: w1[h,f]
    w1t = tl.load(w1_ptr + rh[None, :] * sw1_0 + rf[:, None] * sw1_1,
                  mask=(rh[None, :] < NH) & (rf[:, None] < NF), other=0.0)
    s1 = tl.dot(x, w1t)  # [BM, BH]
    b1 = tl.load(b1_ptr + rh, mask=rh < NH, other=0.0)
    s1 += b1[None, :]

    # adj [BM, BM]
    adj = tl.load(adj_ptr + rm[:, None] * sa0 + rm[None, :] * sa1,
                  mask=(rm[:, None] < M) & (rm[None, :] < M), other=0.0)
    out1 = tl.dot(adj, s1)  # [BM, BH]
    if USE_RELU:
        out1 = tl.where(out1 > 0, out1, tl.exp(out1) - 1.0)

    # w2 [NC, NH] -> w2^T [BH, BC]
    w2t = tl.load(w2_ptr + rc[None, :] * sw2_0 + rh[:, None] * sw2_1,
                  mask=(rc[None, :] < NC) & (rh[:, None] < NH), other=0.0)
    s2 = tl.dot(out1, w2t)  # [BM, BC]
    b2 = tl.load(b2_ptr + rc, mask=rc < NC, other=0.0)
    s2 += b2[None, :]

    out2 = tl.dot(adj, s2)  # [BM, BC]
    tl.store(out_ptr + rm[:, None] * so0 + rc[None, :] * so1, out2,
             mask=(rm[:, None] < M) & (rc[None, :] < NC))


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

    def forward(self, input, adj):
        return None


class GCNNew(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCNNew, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.elu = torch.nn.ELU(inplace=True)

    def forward(self, x, adj, use_relu=True):
        x = x.contiguous()
        adj = adj.contiguous()
        M, NF = x.shape
        NH = self.gc1.W.weight.size(0)
        NC = self.gc2.W.weight.size(0)
        w1 = self.gc1.W.weight
        b1 = self.gc1.W.bias
        w2 = self.gc2.W.weight
        b2 = self.gc2.W.bias

        # Fully-fused single-block path for small graphs (no dropout / eval).
        if (not self.training) and M <= 128 and NF <= 128 and NH <= 128 and NC <= 128:
            def bz(v):
                return max(16, triton.next_power_of_2(v))
            out = torch.empty((M, NC), device=x.device, dtype=torch.float32)
            _fused_kernel[(1,)](
                x, adj, w1, b1, w2, b2, out,
                M, NF, NH, NC,
                x.stride(0), x.stride(1), adj.stride(0), adj.stride(1),
                w1.stride(0), w1.stride(1), w2.stride(0), w2.stride(1),
                out.stride(0), out.stride(1),
                USE_RELU=use_relu,
                BM=bz(M), BF=bz(NF), BH=bz(NH), BC=bz(NC),
                num_warps=1, num_stages=1,
            )
            return out

        # layer 1: support = x @ W1^T + b1
        w1 = self.gc1.W.weight  # [nhid, nfeat]
        b1 = self.gc1.W.bias
        # B = W1^T -> pass weight with swapped strides
        support1 = _mm(x, w1.t(), bias=b1, elu=False)
        # output1 = adj @ support1, with optional ELU epilogue
        out1 = _mm(adj, support1, bias=None, elu=use_relu)
        # dropout (eval -> identity; training -> use torch for randomness parity)
        out1 = F.dropout(out1, self.dropout, training=self.training)
        # layer 2
        w2 = self.gc2.W.weight
        b2 = self.gc2.W.bias
        support2 = _mm(out1, w2.t(), bias=b2, elu=False)
        out2 = _mm(adj, support2, bias=None, elu=False)
        return out2
