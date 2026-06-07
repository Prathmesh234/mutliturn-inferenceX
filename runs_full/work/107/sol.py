import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_kernel(x_ptr, w_ptr, b_ptr, y_ptr, M, N, K,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N
    k_mask = offs_k < K
    # x: (BLOCK_M, BLOCK_K)
    x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=m_mask[:, None] & k_mask[None, :], other=0.0)
    # w: (BLOCK_N, BLOCK_K)  (nn.Linear weight is (N, K))
    w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :],
                mask=n_mask[:, None] & k_mask[None, :], other=0.0)
    # acc (BLOCK_M, BLOCK_N) = sum_k x[m,k]*w[n,k]
    acc = tl.sum(x[:, None, :] * w[None, :, :], axis=2)
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], acc,
             mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr, SEQ, DK, scale,
                 BLOCK_N: tl.constexpr, BLOCK_DK: tl.constexpr):
    pid = tl.program_id(0)
    bh = pid // SEQ
    i = pid % SEQ
    base = bh * SEQ * DK
    offs_d = tl.arange(0, BLOCK_DK)
    offs_n = tl.arange(0, BLOCK_N)
    d_mask = offs_d < DK
    n_mask = offs_n < SEQ
    # q row: (BLOCK_DK,)
    q = tl.load(q_ptr + base + i * DK + offs_d, mask=d_mask, other=0.0)
    # k block: (BLOCK_N, BLOCK_DK)
    k = tl.load(k_ptr + base + offs_n[:, None] * DK + offs_d[None, :],
                mask=n_mask[:, None] & d_mask[None, :], other=0.0)
    scores = tl.sum(q[None, :] * k, axis=1) * scale  # (BLOCK_N,)
    scores = tl.where(n_mask, scores, -float('inf'))
    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    denom = tl.sum(p, axis=0)
    p = p / denom
    # v block: (BLOCK_N, BLOCK_DK)
    v = tl.load(v_ptr + base + offs_n[:, None] * DK + offs_d[None, :],
                mask=n_mask[:, None] & d_mask[None, :], other=0.0)
    out = tl.sum(p[:, None] * v, axis=0)  # (BLOCK_DK,)
    tl.store(o_ptr + base + i * DK + offs_d, out, mask=d_mask)


def _linear(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_M = 64
    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_K = triton.next_power_of_2(K)
    grid = (triton.cdiv(M, BLOCK_M),)
    _linear_kernel[grid](x, weight, bias, y, M, N, K,
                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                         num_warps=4)
    return y


class MultiHeadAttentionNew(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        d_model = self.d_model
        h = self.h
        d_k = self.d_k

        q2 = q.reshape(-1, d_model).contiguous()
        k2 = k.reshape(-1, d_model).contiguous()
        v2 = v.reshape(-1, d_model).contiguous()

        qp = _linear(q2, self.q_linear.weight, self.q_linear.bias)
        kp = _linear(k2, self.k_linear.weight, self.k_linear.bias)
        vp = _linear(v2, self.v_linear.weight, self.v_linear.bias)

        M = qp.shape[0]
        seq = M // bs
        # (bs, seq, h, d_k) -> (bs, h, seq, d_k)
        qh = qp.view(bs, seq, h, d_k).transpose(1, 2).contiguous()
        kh = kp.view(bs, seq, h, d_k).transpose(1, 2).contiguous()
        vh = vp.view(bs, seq, h, d_k).transpose(1, 2).contiguous()

        oh = torch.empty_like(qh)
        scale = 1.0 / math.sqrt(d_k)
        BLOCK_N = triton.next_power_of_2(seq)
        BLOCK_DK = triton.next_power_of_2(d_k)
        grid = (bs * h * seq,)
        _attn_kernel[grid](qh, kh, vh, oh, seq, d_k, scale,
                           BLOCK_N=BLOCK_N, BLOCK_DK=BLOCK_DK, num_warps=4)

        concat = oh.transpose(1, 2).contiguous().view(bs * seq, d_model)
        out = _linear(concat, self.out.weight, self.out.bias)
        return out.view(bs, seq, d_model)
