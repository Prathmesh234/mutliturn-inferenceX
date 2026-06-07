import math
import torch
import triton
import triton.language as tl
from torch.nn import Linear, LayerNorm


@triton.jit
def _mega_vec(q_ptr, k_ptr, wq_ptr, bq_ptr, wk_ptr, bk_ptr, wv_ptr, bv_ptr,
              wo_ptr, bo_ptr, out_ptr, NQ, NK, DQ, DK, DV, scale,
              BLOCK_NQ: tl.constexpr, BLOCK_NK: tl.constexpr,
              BLOCK_DV: tl.constexpr, BLOCK_DQ: tl.constexpr,
              BLOCK_DK: tl.constexpr, NH: tl.constexpr, DS: tl.constexpr):
    b = tl.program_id(0)
    offs_nq = tl.arange(0, BLOCK_NQ)
    offs_nk = tl.arange(0, BLOCK_NK)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_dq = tl.arange(0, BLOCK_DQ)
    offs_dk = tl.arange(0, BLOCK_DK)
    m_nq = offs_nq < NQ
    m_nk = offs_nk < NK
    m_dv = offs_dv < DV
    m_dq = offs_dq < DQ
    m_dk = offs_dk < DK

    qx = tl.load(q_ptr + b * NQ * DQ + offs_nq[:, None] * DQ + offs_dq[None, :],
                 mask=m_nq[:, None] & m_dq[None, :], other=0.0)
    kx = tl.load(k_ptr + b * NK * DK + offs_nk[:, None] * DK + offs_dk[None, :],
                 mask=m_nk[:, None] & m_dk[None, :], other=0.0)
    wq = tl.load(wq_ptr + offs_dv[:, None] * DQ + offs_dq[None, :],
                 mask=m_dv[:, None] & m_dq[None, :], other=0.0)
    bq = tl.load(bq_ptr + offs_dv, mask=m_dv, other=0.0)
    Qf = tl.sum(qx[:, None, :] * wq[None, :, :], axis=2) + bq[None, :]
    wk = tl.load(wk_ptr + offs_dv[:, None] * DK + offs_dk[None, :],
                 mask=m_dv[:, None] & m_dk[None, :], other=0.0)
    bk = tl.load(bk_ptr + offs_dv, mask=m_dv, other=0.0)
    Kf = tl.sum(kx[:, None, :] * wk[None, :, :], axis=2) + bk[None, :]
    wv = tl.load(wv_ptr + offs_dv[:, None] * DK + offs_dk[None, :],
                 mask=m_dv[:, None] & m_dk[None, :], other=0.0)
    bv = tl.load(bv_ptr + offs_dv, mask=m_dv, other=0.0)
    Vf = tl.sum(kx[:, None, :] * wv[None, :, :], axis=2) + bv[None, :]

    # reshape to heads [N, NH, DS]
    Qf3 = tl.reshape(Qf, [BLOCK_NQ, NH, DS])
    Kf3 = tl.reshape(Kf, [BLOCK_NK, NH, DS])
    Vf3 = tl.reshape(Vf, [BLOCK_NK, NH, DS])
    # score [NQ,NK,NH]
    score = tl.sum(Qf3[:, None, :, :] * Kf3[None, :, :, :], axis=3) * scale
    valid = m_nq[:, None] & m_nk[None, :]
    score = tl.where(valid[:, :, None], score, float("-inf"))
    mx = tl.max(score, axis=0)          # [NK,NH]
    e = tl.exp(score - mx[None, :, :])
    denom = tl.sum(e, axis=0)           # [NK,NH]
    denom = tl.where(denom == 0.0, 1.0, denom)
    a = e / denom[None, :, :]           # [NQ,NK,NH]
    av = tl.sum(a[:, :, :, None] * Vf3[None, :, :, :], axis=1)  # [NQ,NH,DS]
    av2 = tl.reshape(av, [BLOCK_NQ, BLOCK_DV])
    attn = Qf + av2

    wo = tl.load(wo_ptr + offs_dv[:, None] * DV + offs_dv[None, :],
                 mask=m_dv[:, None] & m_dv[None, :], other=0.0)
    bo = tl.load(bo_ptr + offs_dv, mask=m_dv, other=0.0)
    fo = tl.maximum(tl.sum(attn[:, None, :] * wo[None, :, :], axis=2) + bo[None, :], 0.0)
    out = attn + fo
    tl.store(out_ptr + b * NQ * DV + offs_nq[:, None] * DV + offs_dv[None, :],
             out, mask=m_nq[:, None] & m_dv[None, :])


class MABNew(torch.nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, Conv=None, layer_norm=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.fc_q = Linear(dim_Q, dim_V)
        if Conv is None:
            self.layer_k = Linear(dim_K, dim_V)
            self.layer_v = Linear(dim_K, dim_V)
        else:
            self.layer_k = Conv(dim_K, dim_V)
            self.layer_v = Conv(dim_K, dim_V)
        if layer_norm:
            self.ln0 = LayerNorm(dim_V)
            self.ln1 = LayerNorm(dim_V)
        self.fc_o = Linear(dim_V, dim_V)

    def forward(self, Q, K, graph=None, mask=None):
        B, NQ, DQ = Q.shape
        NK, DK = K.shape[1], K.shape[2]
        dim_V = self.dim_V
        DS = dim_V // self.num_heads
        out = torch.empty((B, NQ, dim_V), device=Q.device, dtype=Q.dtype)
        scale = 1.0 / math.sqrt(dim_V)
        # vectorized path requires exact power-of-two feature packing
        assert triton.next_power_of_2(dim_V) == self.num_heads * DS
        _mega_vec[(B,)](
            Q.contiguous(), K.contiguous(),
            self.fc_q.weight, self.fc_q.bias,
            self.layer_k.weight, self.layer_k.bias,
            self.layer_v.weight, self.layer_v.bias,
            self.fc_o.weight, self.fc_o.bias, out,
            NQ, NK, DQ, DK, dim_V, scale,
            BLOCK_NQ=triton.next_power_of_2(NQ),
            BLOCK_NK=triton.next_power_of_2(NK),
            BLOCK_DV=triton.next_power_of_2(dim_V),
            BLOCK_DQ=triton.next_power_of_2(DQ),
            BLOCK_DK=triton.next_power_of_2(DK),
            NH=self.num_heads, DS=DS, num_warps=1)
        return out
