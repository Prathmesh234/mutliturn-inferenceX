import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def attn_compute(xq, xk, xv, Wq, Wk, Wv, Wfc, G, Bt,
                 Attn, sa_base, sa_h, sa_q, sa_k,
                 Lq, Lk, temp, eps,
                 swq_r, swq_c, swk_r, swk_c, swv_r, swv_c, swf_r, swf_c,
                 NH: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, DM: tl.constexpr,
                 BL: tl.constexpr, BDK: tl.constexpr, BDV: tl.constexpr, BDM: tl.constexpr):
    offs_l = tl.arange(0, BL)
    offs_c = tl.arange(0, BL)
    rk = tl.arange(0, BDK)
    rv = tl.arange(0, BDV)
    rm = tl.arange(0, BDM)
    fc_acc = tl.zeros((BL, BDM), tl.float32)
    col_ok = offs_c[None, :] < Lk
    for h in range(NH):
        wqh = tl.load(Wq + (h * DK + rk)[:, None] * swq_r + rm[None, :] * swq_c,
                      mask=(rk[:, None] < DK) & (rm[None, :] < DM), other=0.0)
        wkh = tl.load(Wk + (h * DK + rk)[:, None] * swk_r + rm[None, :] * swk_c,
                      mask=(rk[:, None] < DK) & (rm[None, :] < DM), other=0.0)
        wvh = tl.load(Wv + (h * DV + rv)[:, None] * swv_r + rm[None, :] * swv_c,
                      mask=(rv[:, None] < DV) & (rm[None, :] < DM), other=0.0)
        qh = tl.dot(xq, tl.trans(wqh), allow_tf32=False)
        kh = tl.dot(xk, tl.trans(wkh), allow_tf32=False)
        vh = tl.dot(xv, tl.trans(wvh), allow_tf32=False)
        scores = tl.dot(qh, tl.trans(kh), allow_tf32=False) / temp
        scores = tl.where(col_ok, scores, -1e9)
        m = tl.max(scores, axis=1)
        p = tl.exp(scores - m[:, None])
        p = tl.where(col_ok, p, 0.0)
        denom = tl.sum(p, axis=1)
        p = p / denom[:, None]
        tl.store(Attn + sa_base + h * sa_h + offs_l[:, None] * sa_q + offs_c[None, :] * sa_k, p,
                 mask=(offs_l[:, None] < Lq) & (offs_c[None, :] < Lk))
        oh = tl.dot(p, vh, allow_tf32=False)
        wfch = tl.load(Wfc + rm[:, None] * swf_r + (h * DV + rv)[None, :] * swf_c,
                       mask=(rm[:, None] < DM) & (rv[None, :] < DV), other=0.0)
        fc_acc += tl.dot(oh, tl.trans(wfch), allow_tf32=False)
    v_ = fc_acc + xq
    col_m = rm < DM
    mean = tl.sum(tl.where(col_m[None, :], v_, 0.0), axis=1) / DM
    vc = tl.where(col_m[None, :], v_ - mean[:, None], 0.0)
    var = tl.sum(vc * vc, axis=1) / DM
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + rm, mask=col_m, other=0.0)
    bt = tl.load(Bt + rm, mask=col_m, other=0.0)
    return vc * rstd[:, None] * g[None, :] + bt[None, :]


@triton.jit
def decoder_fused(
    Din, Enc,
    Wq1, Wk1, Wv1, Wf1, G1, Bt1,
    Wq2, Wk2, Wv2, Wf2, G2, Bt2,
    W1, B1f, W2, B2f, Gf, Btf,
    Out, AttnS, AttnE,
    L, temp, eps,
    sdi_b, sdi_l, sdi_d, sen_b, sen_l, sen_d,
    sq1r, sq1c, sk1r, sk1c, sv1r, sv1c, sf1r, sf1c,
    sq2r, sq2c, sk2r, sk2c, sv2r, sv2c, sf2r, sf2c,
    sw1r, sw1c, sw2r, sw2c,
    so_b, so_l, so_d,
    sas_b, sas_h, sas_q, sas_k, sae_b, sae_h, sae_q, sae_k,
    NH: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, DM: tl.constexpr, DHID: tl.constexpr,
    BL: tl.constexpr, BDK: tl.constexpr, BDV: tl.constexpr, BDM: tl.constexpr, BHID: tl.constexpr):
    b = tl.program_id(0)
    offs_l = tl.arange(0, BL)
    rm = tl.arange(0, BDM)
    rh = tl.arange(0, BHID)
    lmask = (offs_l[:, None] < L) & (rm[None, :] < DM)
    din = tl.load(Din + b * sdi_b + offs_l[:, None] * sdi_l + rm[None, :] * sdi_d, mask=lmask, other=0.0)
    enc = tl.load(Enc + b * sen_b + offs_l[:, None] * sen_l + rm[None, :] * sen_d, mask=lmask, other=0.0)
    out1 = attn_compute(din, din, din, Wq1, Wk1, Wv1, Wf1, G1, Bt1,
                        AttnS, b * sas_b, sas_h, sas_q, sas_k, L, L, temp, eps,
                        sq1r, sq1c, sk1r, sk1c, sv1r, sv1c, sf1r, sf1c,
                        NH, DK, DV, DM, BL, BDK, BDV, BDM)
    out2 = attn_compute(out1, enc, enc, Wq2, Wk2, Wv2, Wf2, G2, Bt2,
                        AttnE, b * sae_b, sae_h, sae_q, sae_k, L, L, temp, eps,
                        sq2r, sq2c, sk2r, sk2c, sv2r, sv2c, sf2r, sf2c,
                        NH, DK, DV, DM, BL, BDK, BDV, BDM)
    # FFN
    w1 = tl.load(W1 + rh[:, None] * sw1r + rm[None, :] * sw1c,
                 mask=(rh[:, None] < DHID) & (rm[None, :] < DM), other=0.0)
    h = tl.dot(out2, tl.trans(w1), allow_tf32=False)
    b1 = tl.load(B1f + rh, mask=rh < DHID, other=0.0)
    h = tl.maximum(h + b1[None, :], 0.0)
    h = tl.where(rh[None, :] < DHID, h, 0.0)
    w2 = tl.load(W2 + rm[:, None] * sw2r + rh[None, :] * sw2c,
                 mask=(rm[:, None] < DM) & (rh[None, :] < DHID), other=0.0)
    y = tl.dot(h, tl.trans(w2), allow_tf32=False)
    b2 = tl.load(B2f + rm, mask=rm < DM, other=0.0)
    y = y + b2[None, :]
    v_ = y + out2
    col_m = rm < DM
    mean = tl.sum(tl.where(col_m[None, :], v_, 0.0), axis=1) / DM
    vc = tl.where(col_m[None, :], v_ - mean[:, None], 0.0)
    var = tl.sum(vc * vc, axis=1) / DM
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(Gf + rm, mask=col_m, other=0.0)
    bt = tl.load(Btf + rm, mask=col_m, other=0.0)
    out = vc * rstd[:, None] * g[None, :] + bt[None, :]
    tl.store(Out + b * so_b + offs_l[:, None] * so_l + rm[None, :] * so_d, out, mask=lmask)


def _pow2(n):
    return 1 << (max(1, n - 1)).bit_length()


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)


class MultiHeadAttentionNew(nn.Module):
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


class PositionwiseFeedForwardNew(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-06)
        self.dropout = nn.Dropout(dropout)


class DecoderLayerNew(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayerNew, self).__init__()
        self.slf_attn = MultiHeadAttentionNew(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttentionNew(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForwardNew(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        sa = self.slf_attn
        ea = self.enc_attn
        ff = self.pos_ffn
        B, L, DM = dec_input.shape
        NH, DK, DV = sa.n_head, sa.d_k, sa.d_v
        DHID = ff.w_1.weight.shape[0]
        out = torch.empty((B, L, DM), device=dec_input.device, dtype=dec_input.dtype)
        attnS = torch.empty((B, NH, L, L), device=dec_input.device, dtype=dec_input.dtype)
        attnE = torch.empty((B, NH, L, L), device=dec_input.device, dtype=dec_input.dtype)
        BL = max(16, _pow2(L))
        BDK = max(16, _pow2(DK))
        BDV = max(16, _pow2(DV))
        BDM = max(16, _pow2(DM))
        BHID = max(16, _pow2(DHID))
        temp = float(sa.attention.temperature)
        decoder_fused[(B,)](
            dec_input, enc_output,
            sa.w_qs.weight, sa.w_ks.weight, sa.w_vs.weight, sa.fc.weight,
            sa.layer_norm.weight, sa.layer_norm.bias,
            ea.w_qs.weight, ea.w_ks.weight, ea.w_vs.weight, ea.fc.weight,
            ea.layer_norm.weight, ea.layer_norm.bias,
            ff.w_1.weight, ff.w_1.bias, ff.w_2.weight, ff.w_2.bias,
            ff.layer_norm.weight, ff.layer_norm.bias,
            out, attnS, attnE,
            L, temp, 1e-06,
            dec_input.stride(0), dec_input.stride(1), dec_input.stride(2),
            enc_output.stride(0), enc_output.stride(1), enc_output.stride(2),
            sa.w_qs.weight.stride(0), sa.w_qs.weight.stride(1),
            sa.w_ks.weight.stride(0), sa.w_ks.weight.stride(1),
            sa.w_vs.weight.stride(0), sa.w_vs.weight.stride(1),
            sa.fc.weight.stride(0), sa.fc.weight.stride(1),
            ea.w_qs.weight.stride(0), ea.w_qs.weight.stride(1),
            ea.w_ks.weight.stride(0), ea.w_ks.weight.stride(1),
            ea.w_vs.weight.stride(0), ea.w_vs.weight.stride(1),
            ea.fc.weight.stride(0), ea.fc.weight.stride(1),
            ff.w_1.weight.stride(0), ff.w_1.weight.stride(1),
            ff.w_2.weight.stride(0), ff.w_2.weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            attnS.stride(0), attnS.stride(1), attnS.stride(2), attnS.stride(3),
            attnE.stride(0), attnE.stride(1), attnE.stride(2), attnE.stride(3),
            NH=NH, DK=DK, DV=DV, DM=DM, DHID=DHID,
            BL=BL, BDK=BDK, BDV=BDV, BDM=BDM, BHID=BHID, num_warps=4)
        return out, attnS, attnE
