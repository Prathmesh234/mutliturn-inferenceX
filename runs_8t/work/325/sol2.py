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
def _kv_kernel(x_ptr, wk_ptr, bk_ptr, wv_ptr, bv_ptr, k_ptr, v_ptr,
               M, C, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0)
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    kmask = offs_k < C
    nmask = offs_n < C
    x_row = tl.load(x_ptr + m * C + offs_k, mask=kmask, other=0.0)
    wk = tl.load(wk_ptr + offs_n[:, None] * C + offs_k[None, :],
                 mask=nmask[:, None] & kmask[None, :], other=0.0)
    wv = tl.load(wv_ptr + offs_n[:, None] * C + offs_k[None, :],
                 mask=nmask[:, None] & kmask[None, :], other=0.0)
    ak = tl.sum(x_row[None, :] * wk, axis=1) + tl.load(bk_ptr + offs_n, mask=nmask, other=0.0)
    av = tl.sum(x_row[None, :] * wv, axis=1) + tl.load(bv_ptr + offs_n, mask=nmask, other=0.0)
    tl.store(k_ptr + m * C + offs_n, ak, mask=nmask)
    tl.store(v_ptr + m * C + offs_n, av, mask=nmask)


@triton.jit
def _attn_kernel(q_ptr, k_ptr, v_ptr, out_ptr,
                 B, SN, NN, C, H, D, scale,
                 BS: tl.constexpr, BNN: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    offs_s = tl.arange(0, BS)
    offs_n = tl.arange(0, BNN)
    offs_d = tl.arange(0, BD)
    smask = offs_s < SN
    nmask = offs_n < NN
    dmask = offs_d < D
    col = h * D + offs_d
    q = tl.load(q_ptr + offs_s[:, None] * C + col[None, :],
                mask=smask[:, None] & dmask[None, :], other=0.0)
    base = b * NN * C
    k = tl.load(k_ptr + base + offs_n[:, None] * C + col[None, :],
                mask=nmask[:, None] & dmask[None, :], other=0.0)
    v = tl.load(v_ptr + base + offs_n[:, None] * C + col[None, :],
                mask=nmask[:, None] & dmask[None, :], other=0.0)
    scores = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale
    scores = tl.where(smask[:, None], scores, float('-inf'))
    m = tl.max(scores, axis=0)
    e = tl.exp(scores - m[None, :])
    denom = tl.sum(e, axis=0)
    a = e / denom[None, :]
    contrib = tl.sum(a[:, :, None] * v[None, :, :], axis=1)
    out = q + contrib
    tl.store(out_ptr + (b * SN + offs_s[:, None]) * C + col[None, :],
             out, mask=smask[:, None] & dmask[None, :])


def _linear(x2d, weight, bias, res=None):
    M, K = x2d.shape
    Nout = weight.shape[0]
    out = torch.empty((M, Nout), device=x2d.device, dtype=x2d.dtype)
    _linear_kernel[(M,)](x2d, weight, bias, res if res is not None else x2d, out,
                         M, K, Nout, BN=_next_pow2(Nout), BK=_next_pow2(K),
                         APPLY=res is not None, num_warps=4)
    return out


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
        Q = _linear(self.S.reshape(SN, C), mab.fc_q.weight, mab.fc_q.bias)
        xf = x.reshape(B * N, C)
        Kt = torch.empty((B * N, C), device=x.device, dtype=x.dtype)
        Vt = torch.empty((B * N, C), device=x.device, dtype=x.dtype)
        _kv_kernel[(B * N,)](xf, mab.layer_k.weight, mab.layer_k.bias,
                             mab.layer_v.weight, mab.layer_v.bias, Kt, Vt,
                             B * N, C, BN=_next_pow2(C), BK=_next_pow2(C), num_warps=4)
        out0 = torch.empty((B, SN, C), device=x.device, dtype=x.dtype)
        scale = 1.0 / math.sqrt(C)
        _attn_kernel[(B * H,)](Q, Kt, Vt, out0, B, SN, N, C, H, D, scale,
                               BS=_next_pow2(SN), BNN=_next_pow2(N), BD=_next_pow2(D), num_warps=4)
        out = _linear(out0.reshape(B * SN, C), mab.fc_o.weight, mab.fc_o.bias,
                      res=out0.reshape(B * SN, C))
        return out.reshape(B, SN, C)
