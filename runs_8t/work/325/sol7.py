import math
import torch
import triton
import triton.language as tl


def _next_pow2(n):
    return 1 << (max(1, n) - 1).bit_length()


@triton.jit
def _full_kernel(x_ptr, s_ptr, wq_ptr, bq_ptr, wk_ptr, bk_ptr, wv_ptr, bv_ptr,
                 wo_ptr, bo_ptr, out_ptr, B, SN, NN, C, H, D, scale,
                 BB: tl.constexpr, BS: tl.constexpr, BN: tl.constexpr,
                 BH: tl.constexpr, BD: tl.constexpr, BK: tl.constexpr):
    offs_b = tl.arange(0, BB)
    offs_s = tl.arange(0, BS)
    offs_n = tl.arange(0, BN)
    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, BD)
    offs_k = tl.arange(0, BK)
    bmask = offs_b < B
    smask = offs_s < SN
    nmask = offs_n < NN
    hmask = offs_h < H
    dmask = offs_d < D
    kmask = offs_k < C
    col = offs_h[:, None] * D + offs_d[None, :]
    cmask = hmask[:, None] & dmask[None, :]

    wptr = (col[:, :, None] * C + offs_k[None, None, :])
    wm = cmask[:, :, None] & kmask[None, None, :]
    wq = tl.load(wq_ptr + wptr, mask=wm, other=0.0)
    wk = tl.load(wk_ptr + wptr, mask=wm, other=0.0)
    wv = tl.load(wv_ptr + wptr, mask=wm, other=0.0)
    bq = tl.load(bq_ptr + col, mask=cmask, other=0.0)
    bk = tl.load(bk_ptr + col, mask=cmask, other=0.0)
    bv = tl.load(bv_ptr + col, mask=cmask, other=0.0)

    # S: [SN, C]
    sx = tl.load(s_ptr + offs_s[:, None] * C + offs_k[None, :],
                 mask=smask[:, None] & kmask[None, :], other=0.0)
    # x: [B, NN, C] -> [BB, BN, BK]
    xb = tl.load(x_ptr + offs_b[:, None, None] * NN * C
                 + offs_n[None, :, None] * C + offs_k[None, None, :],
                 mask=bmask[:, None, None] & nmask[None, :, None] & kmask[None, None, :],
                 other=0.0)

    # Q3[s,h,d] (batch independent)
    Q3 = tl.sum(sx[:, None, None, :] * wq[None, :, :, :], axis=3) + bq[None, :, :]
    # K3[b,n,h,d], V3
    K3 = tl.sum(xb[:, :, None, None, :] * wk[None, None, :, :, :], axis=4) + bk[None, None, :, :]
    V3 = tl.sum(xb[:, :, None, None, :] * wv[None, None, :, :, :], axis=4) + bv[None, None, :, :]

    # scores[b,s,n,h] = sum_d Q3[s,h,d]*K3[b,n,h,d]*scale
    scores = tl.sum(Q3[None, :, None, :, :] * K3[:, None, :, :, :], axis=4) * scale
    scores = tl.where(smask[None, :, None, None], scores, float('-inf'))
    mx = tl.max(scores, axis=1)                       # [BB,BN,BH]
    e = tl.exp(scores - mx[:, None, :, :])
    denom = tl.sum(e, axis=1)
    A = e / denom[:, None, :, :]                       # [BB,BS,BN,BH]

    # out3[b,s,h,d] = Q3[s,h,d] + sum_n A[b,s,n,h]*V3[b,n,h,d]
    out3 = Q3[None, :, :, :] + tl.sum(A[:, :, :, :, None] * V3[:, None, :, :, :], axis=2)

    # fc_o
    wo = tl.load(wo_ptr + col[:, :, None, None] * C + col[None, None, :, :],
                 mask=cmask[:, :, None, None] & cmask[None, None, :, :], other=0.0)
    bo = tl.load(bo_ptr + col, mask=cmask, other=0.0)
    prod = out3[:, :, None, None, :, :] * wo[None, None, :, :, :, :]  # [BB,BS,BHo,BDo,BH,BD]
    t = tl.sum(prod, axis=5)
    outacc = tl.sum(t, axis=4) + bo[None, None, :, :]
    outF = out3 + tl.maximum(outacc, 0.0)

    optr = offs_b[:, None, None, None] * SN * C + offs_s[None, :, None, None] * C + col[None, None, :, :]
    om = bmask[:, None, None, None] & smask[None, :, None, None] & cmask[None, None, :, :]
    tl.store(out_ptr + optr, outF, mask=om)


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
        out = torch.empty((B, SN, C), device=x.device, dtype=x.dtype)
        scale = 1.0 / math.sqrt(C)
        _full_kernel[(1,)](
            x, self.S, mab.fc_q.weight, mab.fc_q.bias,
            mab.layer_k.weight, mab.layer_k.bias,
            mab.layer_v.weight, mab.layer_v.bias,
            mab.fc_o.weight, mab.fc_o.bias, out,
            B, SN, N, C, H, D, scale,
            BB=_next_pow2(B), BS=_next_pow2(SN), BN=_next_pow2(N), BH=_next_pow2(H),
            BD=_next_pow2(D), BK=_next_pow2(C), num_warps=4)
        return out
