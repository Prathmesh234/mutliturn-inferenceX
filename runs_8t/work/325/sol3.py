import math
import torch
import triton
import triton.language as tl


def _next_pow2(n):
    return 1 << (max(1, n) - 1).bit_length()


@triton.jit
def _linear_kernel(x_ptr, w_ptr, b_ptr, res_ptr, out_ptr,
                   M, K, Nout,
                   BN: tl.constexpr, BK: tl.constexpr, APPLY: tl.constexpr):
    m = tl.program_id(0)
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    kmask = offs_k < K
    nmask = offs_n < Nout
    x_row = tl.load(x_ptr + m * K + offs_k, mask=kmask, other=0.0)
    w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :],
                mask=nmask[:, None] & kmask[None, :], other=0.0)
    acc = tl.sum(x_row[None, :] * w, axis=1)
    acc += tl.load(b_ptr + offs_n, mask=nmask, other=0.0)
    if APPLY:
        res = tl.load(res_ptr + m * Nout + offs_n, mask=nmask, other=0.0)
        acc = res + tl.maximum(acc, 0.0)
    tl.store(out_ptr + m * Nout + offs_n, acc, mask=nmask)


@triton.jit
def _mega_kernel(x_ptr, s_ptr, wq_ptr, bq_ptr, wk_ptr, bk_ptr, wv_ptr, bv_ptr,
                 out_ptr, B, SN, NN, C, H, D, scale,
                 BS: tl.constexpr, BN: tl.constexpr, BH: tl.constexpr,
                 BD: tl.constexpr, BK: tl.constexpr):
    b = tl.program_id(0)
    offs_s = tl.arange(0, BS)
    offs_n = tl.arange(0, BN)
    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, BD)
    offs_k = tl.arange(0, BK)
    smask = offs_s < SN
    nmask = offs_n < NN
    hmask = offs_h < H
    dmask = offs_d < D
    kmask = offs_k < C
    col = offs_h[:, None] * D + offs_d[None, :]            # [BH, BD] channel idx
    cmask = hmask[:, None] & dmask[None, :]

    # weights as [BH, BD, BK]
    wptr = (col[:, :, None] * C + offs_k[None, None, :])
    wm = cmask[:, :, None] & kmask[None, None, :]
    wq = tl.load(wq_ptr + wptr, mask=wm, other=0.0)
    wk = tl.load(wk_ptr + wptr, mask=wm, other=0.0)
    wv = tl.load(wv_ptr + wptr, mask=wm, other=0.0)
    bq = tl.load(bq_ptr + col, mask=cmask, other=0.0)
    bk = tl.load(bk_ptr + col, mask=cmask, other=0.0)
    bv = tl.load(bv_ptr + col, mask=cmask, other=0.0)

    # S: [SN, C] -> rows [BS, BK]
    sx = tl.load(s_ptr + offs_s[:, None] * C + offs_k[None, :],
                 mask=smask[:, None] & kmask[None, :], other=0.0)
    # x[b]: [NN, C] -> [BN, BK]
    xb = tl.load(x_ptr + b * NN * C + offs_n[:, None] * C + offs_k[None, :],
                 mask=nmask[:, None] & kmask[None, :], other=0.0)

    # Q3[s,h,d] = sum_k sx[s,k]*wq[h,d,k] + bq[h,d]
    Q3 = tl.sum(sx[:, None, None, :] * wq[None, :, :, :], axis=3) + bq[None, :, :]
    K3 = tl.sum(xb[:, None, None, :] * wk[None, :, :, :], axis=3) + bk[None, :, :]
    V3 = tl.sum(xb[:, None, None, :] * wv[None, :, :, :], axis=3) + bv[None, :, :]

    # scores[s,n,h] = sum_d Q3[s,h,d]*K3[n,h,d]*scale
    scores = tl.sum(Q3[:, None, :, :] * K3[None, :, :, :], axis=3) * scale  # [BS,BN,BH]
    scores = tl.where(smask[:, None, None], scores, float('-inf'))
    mx = tl.max(scores, axis=0)                          # [BN, BH]
    e = tl.exp(scores - mx[None, :, :])
    denom = tl.sum(e, axis=0)                            # [BN, BH]
    A = e / denom[None, :, :]                            # [BS, BN, BH]

    # out3[s,h,d] = Q3 + sum_n A[s,n,h]*V3[n,h,d]
    out3 = Q3 + tl.sum(A[:, :, :, None] * V3[None, :, :, :], axis=1)  # [BS,BH,BD]

    # store to out0[b, s, h*D+d]
    optr = b * SN * C + offs_s[:, None, None] * C + col[None, :, :]
    om = smask[:, None, None] & cmask[None, :, :]
    tl.store(out_ptr + optr, out3, mask=om)


class PMANew(torch.nn.Module):
    def __init__(self, channels, num_heads, num_seeds, Conv=None, layer_norm=False):
        super().__init__()
        from torch.nn import Linear, LayerNorm
        self.S = torch.nn.Parameter(torch.Tensor(1, num_seeds, channels))

        class _MAB(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.dim_V = channels
                s.num_heads = num_heads
                s.layer_norm = layer_norm
                s.fc_q = Linear(channels, channels)
                s.layer_k = Linear(channels, channels)
                s.layer_v = Linear(channels, channels)
                if layer_norm:
                    s.ln0 = LayerNorm(channels)
                    s.ln1 = LayerNorm(channels)
                s.fc_o = Linear(channels, channels)
        self.mab = _MAB()
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        torch.nn.init.xavier_uniform_(self.S)

    def forward(self, x, graph=None, mask=None):
        assert graph is None and mask is None and not self.layer_norm
        B, N, C = x.shape
        SN = self.S.shape[1]
        H = self.num_heads
        D = C // H
        mab = self.mab
        out0 = torch.empty((B, SN, C), device=x.device, dtype=x.dtype)
        scale = 1.0 / math.sqrt(C)
        _mega_kernel[(B,)](
            x, self.S, mab.fc_q.weight, mab.fc_q.bias,
            mab.layer_k.weight, mab.layer_k.bias,
            mab.layer_v.weight, mab.layer_v.bias, out0,
            B, SN, N, C, H, D, scale,
            BS=_next_pow2(SN), BN=_next_pow2(N), BH=_next_pow2(H),
            BD=_next_pow2(D), BK=_next_pow2(C), num_warps=4)
        o2 = out0.reshape(B * SN, C)
        out = torch.empty_like(o2)
        _linear_kernel[(B * SN,)](o2, mab.fc_o.weight, mab.fc_o.bias, o2, out,
                                  B * SN, C, C, BN=_next_pow2(C), BK=_next_pow2(C),
                                  APPLY=True, num_warps=4)
        return out.reshape(B, SN, C)
