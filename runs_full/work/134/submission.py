import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_lin(x_ptr, wl_ptr, bl_ptr, wa_ptr, ba_ptr, ww_ptr, bw_ptr,
               f_ptr, ar_ptr, Hin, Hout,
               BK: tl.constexpr, BH: tl.constexpr):
    pid = tl.program_id(0)
    k = tl.arange(0, BK)
    kmask = k < Hin
    x = tl.load(x_ptr + pid * Hin + k, mask=kmask, other=0.0)
    e = tl.arange(0, BH)
    emask = e < Hout
    Wl = tl.load(wl_ptr + e[:, None] * Hin + k[None, :],
                 mask=emask[:, None] & kmask[None, :], other=0.0)
    f_row = tl.sum(Wl * x[None, :], axis=1) + tl.load(bl_ptr + e, mask=emask, other=0.0)
    tl.store(f_ptr + pid * Hout + e, f_row, mask=emask)
    j = tl.arange(0, BH)
    jmask = j < Hout
    Wa = tl.load(wa_ptr + j[:, None] * Hout + e[None, :],
                 mask=jmask[:, None] & emask[None, :], other=0.0)
    h = tl.sum(Wa * f_row[None, :], axis=1) + tl.load(ba_ptr + j, mask=jmask, other=0.0)
    h = tl.maximum(h, 0.0)
    Ww = tl.load(ww_ptr + j, mask=jmask, other=0.0)
    ar = tl.sum(Ww * h, axis=0) + tl.load(bw_ptr)
    tl.store(ar_ptr + pid, ar)


@triton.jit
def _pool(f_ptr, ar_ptr, mask_ptr, out_ptr,
          D0, D1, D2, Hout, BD2: tl.constexpr, BH: tl.constexpr):
    pid = tl.program_id(0)
    a = pid // (D1 * D2)
    rem = pid % (D1 * D2)
    c = rem // D2
    d = rem % D2
    e = tl.arange(0, BH)
    emask = e < Hout
    dp = tl.arange(0, BD2)
    dpmask = dp < D2
    acc_sap = tl.zeros([BH], tl.float32)
    acc_var = tl.zeros([BH], tl.float32)
    for b in range(0, D0):
        mask_off = a * (D0 * D1 * D2) + b * (D1 * D2) + c * D2 + dp
        mval = tl.load(mask_ptr + mask_off, mask=dpmask, other=0.0)
        ar_off = b * (D1 * D2) + c * D2 + dp
        arval = tl.load(ar_ptr + ar_off, mask=dpmask, other=0.0)
        logits = tl.where(dpmask, mval + arval, -float('inf'))
        m = tl.max(logits, axis=0)
        expv = tl.exp(logits - m)
        denom = tl.sum(expv, axis=0)
        w = tl.sum(tl.where(dp == d, expv, 0.0), axis=0) / denom
        f_off = b * (D1 * D2 * Hout) + c * (D2 * Hout) + d * Hout + e
        fb = tl.load(f_ptr + f_off, mask=emask, other=0.0)
        acc_sap += w * fb
        acc_var += w * fb * fb
    variance = tl.sqrt(acc_var - acc_sap * acc_sap + 1e-8)
    out_base = a * (D1 * D2 * 2 * Hout) + c * (D2 * 2 * Hout) + d * (2 * Hout)
    tl.store(out_ptr + out_base + e, acc_sap, mask=emask)
    tl.store(out_ptr + out_base + Hout + e, variance, mask=emask)


class AttentivePooling(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super(AttentivePooling, self).__init__()
        self.W_a = nn.Linear(input_dim, input_dim)
        self.W = nn.Linear(input_dim, 1)
        self.act_fn = nn.ReLU()
        self.softmax = nn.functional.softmax


class ASPNew(nn.Module):
    def __init__(self, out_dim, input_dim):
        super(ASPNew, self).__init__()
        self.linear = nn.Linear(input_dim, out_dim)
        self.ap_layer = AttentivePooling(out_dim)

    def forward(self, feature_BxTxH, att_mask_BxT):
        feat = feature_BxTxH.contiguous()
        am = att_mask_BxT.contiguous()
        D0, D1, D2, Hin = feat.shape
        Hout = self.linear.weight.shape[0]
        N = D0 * D1 * D2
        x2d = feat.view(N, Hin)
        f = torch.empty((D0, D1, D2, Hout), device=feat.device, dtype=feat.dtype)
        ar = torch.empty((D0, D1, D2), device=feat.device, dtype=feat.dtype)
        BK = triton.next_power_of_2(Hin)
        BH = triton.next_power_of_2(Hout)
        _fused_lin[(N,)](
            x2d, self.linear.weight, self.linear.bias,
            self.ap_layer.W_a.weight, self.ap_layer.W_a.bias,
            self.ap_layer.W.weight, self.ap_layer.W.bias,
            f, ar, Hin, Hout, BK=BK, BH=BH, num_warps=4)
        M0 = am.shape[0]
        out = torch.empty((M0, D1, D2, 2 * Hout), device=feat.device, dtype=feat.dtype)
        BD2 = triton.next_power_of_2(D2)
        _pool[(M0 * D1 * D2,)](
            f, ar, am, out, D0, D1, D2, Hout, BD2=BD2, BH=BH, num_warps=4)
        return out
