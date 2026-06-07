import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _full_kernel(xq, xk, xv, wq, wk, wv, wo, bq, bk, bv, bo, out_ptr,
                 H, SEQ, DK, DM, scale,
                 BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DK: tl.constexpr,
                 P: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // SEQ
    i = pid % SEQ
    offs_p = tl.arange(0, P)
    offs_n = tl.arange(0, BLOCK_N)
    p_mask = offs_p < DM
    n_mask = offs_n < SEQ
    wqm = tl.load(wq + offs_p[:, None] * DM + offs_p[None, :], mask=p_mask[:, None] & p_mask[None, :], other=0.0)
    wkm = tl.load(wk + offs_p[:, None] * DM + offs_p[None, :], mask=p_mask[:, None] & p_mask[None, :], other=0.0)
    wvm = tl.load(wv + offs_p[:, None] * DM + offs_p[None, :], mask=p_mask[:, None] & p_mask[None, :], other=0.0)
    bqv = tl.load(bq + offs_p, mask=p_mask, other=0.0)
    bkv = tl.load(bk + offs_p, mask=p_mask, other=0.0)
    bvv = tl.load(bv + offs_p, mask=p_mask, other=0.0)
    xqr = tl.load(xq + (b * SEQ + i) * DM + offs_p, mask=p_mask, other=0.0)
    qp = tl.sum(xqr[None, :] * wqm, axis=1) + bqv
    q = tl.reshape(qp, (BLOCK_H, BLOCK_DK))
    xrows = (b * SEQ + offs_n[:, None]) * DM + offs_p[None, :]
    rmask = n_mask[:, None] & p_mask[None, :]
    xkr = tl.load(xk + xrows, mask=rmask, other=0.0)
    xvr = tl.load(xv + xrows, mask=rmask, other=0.0)
    kp = tl.sum(xkr[:, None, :] * wkm[None, :, :], axis=2) + bkv[None, :]
    vp = tl.sum(xvr[:, None, :] * wvm[None, :, :], axis=2) + bvv[None, :]
    k = tl.reshape(kp, (BLOCK_N, BLOCK_H, BLOCK_DK))
    v = tl.reshape(vp, (BLOCK_N, BLOCK_H, BLOCK_DK))
    s = tl.sum(q[None, :, :] * k, axis=2)
    scores = tl.trans(s) * scale
    scores = tl.where(n_mask[None, :], scores, -float('inf'))
    mx = tl.max(scores, axis=1)[:, None]
    pw = tl.exp(scores - mx)
    pw = pw / tl.sum(pw, axis=1)[:, None]
    pt = tl.trans(pw)
    o_hd = tl.sum(pt[:, :, None] * v, axis=0)
    concat = tl.reshape(o_hd, (P,))
    wom = tl.load(wo + offs_p[:, None] * DM + offs_p[None, :], mask=p_mask[:, None] & p_mask[None, :], other=0.0)
    bov = tl.load(bo + offs_p, mask=p_mask, other=0.0)
    res = tl.sum(concat[None, :] * wom, axis=1) + bov
    tl.store(out_ptr + (b * SEQ + i) * DM + offs_p, res, mask=p_mask)


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
        M = q2.shape[0]
        seq = M // bs
        out = torch.empty((M, d_model), device=q2.device, dtype=q2.dtype)
        scale = 1.0 / math.sqrt(d_k)
        BLOCK_H = triton.next_power_of_2(h)
        BLOCK_DK = triton.next_power_of_2(d_k)
        BLOCK_N = triton.next_power_of_2(seq)
        P = BLOCK_H * BLOCK_DK
        _full_kernel[(bs * seq,)](
            q2, k2, v2,
            self.q_linear.weight, self.k_linear.weight, self.v_linear.weight, self.out.weight,
            self.q_linear.bias, self.k_linear.bias, self.v_linear.bias, self.out.bias,
            out, h, seq, d_k, d_model, scale,
            BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N, BLOCK_DK=BLOCK_DK, P=P, num_warps=1)
        return out.view(bs, seq, d_model)
