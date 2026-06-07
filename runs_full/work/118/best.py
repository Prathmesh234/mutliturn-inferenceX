import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _sdpa_kernel(q_ptr, k_ptr, v_ptr, sm_ptr, out_ptr,
                 t_q, t_k, dim, dim_v, scale, causal,
                 stride_qb, stride_qt, stride_qd,
                 stride_kb, stride_kt, stride_kd,
                 stride_vb, stride_vt, stride_vd,
                 stride_sb, stride_st, stride_sk,
                 stride_ob, stride_ot, stride_od,
                 BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_DV: tl.constexpr):
    pid = tl.program_id(0)
    bi = pid // t_q
    qi = pid % t_q

    offs_d = tl.arange(0, BLOCK_D)
    offs_t = tl.arange(0, BLOCK_T)
    offs_dv = tl.arange(0, BLOCK_DV)

    q = tl.load(q_ptr + bi * stride_qb + qi * stride_qt + offs_d * stride_qd,
                mask=offs_d < dim, other=0.0)

    k_ptrs = k_ptr + bi * stride_kb + offs_t[:, None] * stride_kt + offs_d[None, :] * stride_kd
    k = tl.load(k_ptrs, mask=(offs_t[:, None] < t_k) & (offs_d[None, :] < dim), other=0.0)

    scores = tl.sum(k * q[None, :], axis=1) * scale

    valid = offs_t < t_k
    if causal:
        valid = valid & (offs_t <= qi)
    scores = tl.where(valid, scores, -1e30)

    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    p = tl.where(valid, p, 0.0)
    denom = tl.sum(p, axis=0)
    p = p / denom

    tl.store(sm_ptr + bi * stride_sb + qi * stride_st + offs_t * stride_sk,
             p, mask=offs_t < t_k)

    v_ptrs = v_ptr + bi * stride_vb + offs_t[:, None] * stride_vt + offs_dv[None, :] * stride_vd
    v = tl.load(v_ptrs, mask=(offs_t[:, None] < t_k) & (offs_dv[None, :] < dim_v), other=0.0)

    out = tl.sum(p[:, None] * v, axis=0)
    tl.store(out_ptr + bi * stride_ob + qi * stride_ot + offs_dv * stride_od,
             out, mask=offs_dv < dim_v)


def _next_pow2(x):
    return 1 << (max(x, 1) - 1).bit_length()


class SDPAttentionNew(nn.Module):
    def __init__(self, dropout=0, causal=False):
        super(SDPAttentionNew, self).__init__()
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
        b_v, t_v, dim_v = list(v.size())
        assert b_q == b_k and b_k == b_v
        assert dim_q == dim_k
        assert t_k == t_v
        b = b_q

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        sm_qk = torch.empty((b, t_q, t_k), device=q.device, dtype=q.dtype)
        out = torch.empty((b, t_q, dim_v), device=q.device, dtype=q.dtype)

        scale = 1.0 / (dim_k ** 0.5)
        BLOCK_T = _next_pow2(t_k)
        BLOCK_D = _next_pow2(dim_k)
        BLOCK_DV = _next_pow2(dim_v)

        grid = (b * t_q,)
        _sdpa_kernel[grid](
            q, k, v, sm_qk, out,
            t_q, t_k, dim_k, dim_v, scale, 1 if self.causal else 0,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            sm_qk.stride(0), sm_qk.stride(1), sm_qk.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D, BLOCK_DV=BLOCK_DV,
            num_warps=1,
        )
        sm_qk = self.dropout(sm_qk)
        return out, sm_qk
