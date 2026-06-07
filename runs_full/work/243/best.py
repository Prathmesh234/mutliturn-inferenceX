import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _attn_kernel(Q, K, V, P, O, N, D, scale,
                 stride_b, stride_n, stride_d,
                 BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // N
    i = pid % N
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    mask_d = offs_d < D
    mask_n = offs_n < N

    q = tl.load(Q + b * stride_b + i * stride_n + offs_d * stride_d,
                mask=mask_d, other=0.0)
    k = tl.load(K + b * stride_b + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
                mask=mask_n[:, None] & mask_d[None, :], other=0.0)
    scores = tl.sum(q[None, :] * k, axis=1) * scale
    scores = tl.where(mask_n, scores, -float('inf'))
    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    p = p / tl.sum(p, axis=0)
    tl.store(P + b * N * N + i * N + offs_n, p, mask=mask_n)

    v = tl.load(V + b * stride_b + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
                mask=mask_n[:, None] & mask_d[None, :], other=0.0)
    o = tl.sum(p[:, None] * v, axis=0)
    tl.store(O + b * stride_b + i * stride_n + offs_d * stride_d, o, mask=mask_d)


class GroverAttentionNew(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        d = query.size(-1)
        N = query.size(-2)
        scale = 1.0 / math.sqrt(d)

        q = query.contiguous()
        k = key.contiguous()
        v = value.contiguous()
        B = q.numel() // (N * d)

        qf = q.view(B, N, d)
        kf = k.view(B, N, d)
        vf = v.view(B, N, d)

        if mask is not None:
            scores = torch.matmul(query, key.transpose(-2, -1)) * scale
            scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = torch.softmax(scores, dim=-1)
            if dropout is not None:
                p_attn = dropout(p_attn)
            return torch.matmul(p_attn, value), p_attn

        out = torch.empty_like(qf)
        p_out = torch.empty((B, N, N), device=q.device, dtype=q.dtype)

        BLOCK_N = triton.next_power_of_2(N)
        BLOCK_D = triton.next_power_of_2(d)
        grid = (B * N,)
        _attn_kernel[grid](qf, kf, vf, p_out, out, N, d, scale,
                           qf.stride(0), qf.stride(1), qf.stride(2),
                           BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, num_warps=4)

        out = out.view(query.shape)
        p_attn = p_out.view(*query.shape[:-2], N, N)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return out, p_attn
