import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _encoder_kernel(X, Wq, Wk, Wv, Wfc, W1, b1, W2, b2,
                    ln1w, ln1b, ln2w, ln2b, OUT, ATTN,
                    temp, eps,
                    LEN: tl.constexpr, DM: tl.constexpr, NH: tl.constexpr,
                    DK: tl.constexpr, DV: tl.constexpr, DH: tl.constexpr):
    b = tl.program_id(0)
    li = tl.arange(0, LEN)
    di = tl.arange(0, DM)
    ki = tl.arange(0, DK)
    vi = tl.arange(0, DV)
    ji = tl.arange(0, DH)

    # load x[LEN, DM]
    x = tl.load(X + b * LEN * DM + li[:, None] * DM + di[None, :])

    o_acc = tl.zeros((LEN, DM), tl.float32)
    for h in tl.static_range(NH):
        wq = tl.load(Wq + (h * DK + ki)[:, None] * DM + di[None, :])  # [DK,DM]
        wk = tl.load(Wk + (h * DK + ki)[:, None] * DM + di[None, :])
        wv = tl.load(Wv + (h * DV + vi)[:, None] * DM + di[None, :])  # [DV,DM]
        # q[LEN,DK] = x @ wq^T
        q = tl.sum(x[:, None, :] * wq[None, :, :], axis=2)
        k = tl.sum(x[:, None, :] * wk[None, :, :], axis=2)
        v = tl.sum(x[:, None, :] * wv[None, :, :], axis=2)  # [LEN,DV]
        # attn[LEN,LEN] = (q/temp) @ k^T
        attn = tl.sum((q / temp)[:, None, :] * k[None, :, :], axis=2)
        attn = attn - tl.max(attn, axis=1)[:, None]
        e = tl.exp(attn)
        attn = e / tl.sum(e, axis=1)[:, None]
        tl.store(ATTN + b * NH * LEN * LEN + h * LEN * LEN +
                 li[:, None] * LEN + li[None, :], attn)
        # out_h[LEN,DV] = attn @ v
        outh = tl.sum(attn[:, :, None] * v[None, :, :], axis=1)
        # fc contribution: wfc_h[DM,DV] = Wfc[:, h*DV: h*DV+DV]
        wfc = tl.load(Wfc + di[:, None] * (NH * DV) + (h * DV + vi)[None, :])
        o_acc += tl.sum(outh[:, None, :] * wfc[None, :, :], axis=2)

    # add residual + layernorm1 (over DM)
    o_acc += x
    mean = tl.sum(o_acc, axis=1)[:, None] / DM
    xc = o_acc - mean
    var = tl.sum(xc * xc, axis=1)[:, None] / DM
    h1 = xc / tl.sqrt(var + eps) * tl.load(ln1w + di)[None, :] + tl.load(ln1b + di)[None, :]

    # ffn
    w1 = tl.load(W1 + ji[:, None] * DM + di[None, :])  # [DH,DM]
    t = tl.sum(h1[:, None, :] * w1[None, :, :], axis=2) + tl.load(b1 + ji)[None, :]
    t = tl.maximum(t, 0.0)
    w2 = tl.load(W2 + di[:, None] * DH + ji[None, :])  # [DM,DH]
    o2 = tl.sum(t[:, None, :] * w2[None, :, :], axis=2) + tl.load(b2 + di)[None, :]

    o2 += h1
    mean2 = tl.sum(o2, axis=1)[:, None] / DM
    xc2 = o2 - mean2
    var2 = tl.sum(xc2 * xc2, axis=1)[:, None] / DM
    out = xc2 / tl.sqrt(var2 + eps) * tl.load(ln2w + di)[None, :] + tl.load(ln2b + di)[None, :]

    tl.store(OUT + b * LEN * DM + li[:, None] * DM + di[None, :], out)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-06)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-06)
        self.dropout = nn.Dropout(dropout)


class EncoderLayerNew(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        a = self.slf_attn
        p = self.pos_ffn
        sz_b, length, d_model = enc_input.shape
        n_head, d_k, d_v = a.n_head, a.d_k, a.d_v
        d_inner = p.w_1.weight.shape[0]
        x = enc_input.contiguous()
        out = torch.empty_like(x)
        attn = torch.empty((sz_b, n_head, length, length), device=x.device, dtype=x.dtype)
        _encoder_kernel[(sz_b,)](
            x, a.w_qs.weight, a.w_ks.weight, a.w_vs.weight, a.fc.weight,
            p.w_1.weight, p.w_1.bias, p.w_2.weight, p.w_2.bias,
            a.layer_norm.weight, a.layer_norm.bias,
            p.layer_norm.weight, p.layer_norm.bias,
            out, attn, float(d_k ** 0.5), 1e-06,
            LEN=length, DM=d_model, NH=n_head, DK=d_k, DV=d_v, DH=d_inner,
            num_warps=1)
        return out, attn
