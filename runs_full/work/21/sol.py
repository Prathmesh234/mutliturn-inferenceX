import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr, a_ptr,
                 scale, N, D,
                 stride_qb, stride_qn, stride_qd,
                 stride_ab, stride_an, stride_am,
                 BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    bh = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = q_ptr + bh * stride_qb
    k_base = k_ptr + bh * stride_qb
    v_base = v_ptr + bh * stride_qb

    mask_n = offs_n < N
    mask_d = offs_d < D

    q = tl.load(q_base + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd,
                mask=mask_n[:, None] & mask_d[None, :], other=0.0)
    k = tl.load(k_base + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd,
                mask=mask_n[:, None] & mask_d[None, :], other=0.0)
    v = tl.load(v_base + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd,
                mask=mask_n[:, None] & mask_d[None, :], other=0.0)

    q = q * scale
    scores = tl.dot(q, tl.trans(k))  # (BLOCK_N, BLOCK_N)
    col_mask = offs_n[None, :] < N
    scores = tl.where(col_mask, scores, -float('inf'))

    m = tl.max(scores, axis=1)
    p = tl.exp(scores - m[:, None])
    s = tl.sum(p, axis=1)
    attn = p / s[:, None]

    out = tl.dot(attn.to(v.dtype), v)  # (BLOCK_N, BLOCK_D)

    a_base = a_ptr + bh * stride_ab
    tl.store(a_base + offs_n[:, None] * stride_an + offs_n[None, :] * stride_am,
             attn, mask=mask_n[:, None] & (offs_n[None, :] < N))

    o_base = o_ptr + bh * stride_qb
    tl.store(o_base + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd,
             out.to(o_ptr.dtype.element_ty),
             mask=mask_n[:, None] & mask_d[None, :])


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


class ScaledDotProductAttentionNew(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        if mask is not None:
            # fall back to reference path for masked case
            attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
            attn = attn.masked_fill(mask == 0, -1000000000.0)
            attn = self.dropout(torch.softmax(attn, dim=-1))
            output = torch.matmul(attn, v)
            return output, attn

        B, H, N, D = q.shape
        nbh = B * H
        qc = q.contiguous().view(nbh, N, D)
        kc = k.contiguous().view(nbh, N, D)
        vc = v.contiguous().view(nbh, N, D)
        out = torch.empty_like(qc)
        attn = torch.empty((nbh, N, N), device=q.device, dtype=q.dtype)

        BLOCK_N = max(16, _next_pow2(N))
        BLOCK_D = max(16, _next_pow2(D))

        grid = (nbh,)
        _attn_kernel[grid](
            qc, kc, vc, out, attn,
            1.0 / self.temperature, N, D,
            qc.stride(0), qc.stride(1), qc.stride(2),
            attn.stride(0), attn.stride(1), attn.stride(2),
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )

        output = out.view(B, H, N, D)
        attn = self.dropout(attn.view(B, H, N, N))
        return output, attn
