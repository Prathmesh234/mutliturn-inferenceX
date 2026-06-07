import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _feat_s_kernel(x_ptr, wl_ptr, bl_ptr, wa_ptr, ba_ptr, ww_ptr, bw_ptr,
                   feat_ptr, s_ptr, M, Kin, D,
                   BK: tl.constexpr, BD: tl.constexpr):
    m = tl.program_id(0)
    ok = tl.arange(0, BK)
    mk = ok < Kin
    xin = tl.load(x_ptr + m * Kin + ok, mask=mk, other=0.0)
    oj = tl.arange(0, BD)
    mj = oj < D
    wl = tl.load(wl_ptr + oj[:, None] * Kin + ok[None, :],
                 mask=mj[:, None] & mk[None, :], other=0.0)
    feat = tl.sum(wl * xin[None, :], axis=1) + tl.load(bl_ptr + oj, mask=mj, other=0.0)
    tl.store(feat_ptr + m * D + oj, feat, mask=mj)
    wa = tl.load(wa_ptr + oj[:, None] * D + oj[None, :],
                 mask=mj[:, None] & mj[None, :], other=0.0)
    hidden = tl.sum(wa * feat[None, :], axis=1) + tl.load(ba_ptr + oj, mask=mj, other=0.0)
    hidden = tl.maximum(hidden, 0.0)
    ww = tl.load(ww_ptr + oj, mask=mj, other=0.0)
    s = tl.sum(ww * hidden) + tl.load(bw_ptr)
    tl.store(s_ptr + m, s)


@triton.jit
def _smax_wsum_kernel(feat_ptr, mask_ptr, s_ptr, o_ptr, n0, n1, n2, n3, D,
                      BB: tl.constexpr, BL: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0)  # over (a,c,d), shape (n0,n1,n2)
    a = pid // (n1 * n2)
    r = pid % (n1 * n2)
    c = r // n2
    d = r % n2
    ob = tl.arange(0, BB)
    ol = tl.arange(0, BL)
    oe = tl.arange(0, BD)
    mb = ob < n1
    ml = ol < n3
    me = oe < D
    full = mb[:, None] & ml[None, :]
    moff = a * (n1 * n2 * n3) + ob[:, None] * (n2 * n3) + c * n3 + ol[None, :]
    soff = ob[:, None] * (n1 * n2) + c * n2 + ol[None, :]
    mv = tl.load(mask_ptr + moff, mask=full, other=0.0)
    sv = tl.load(s_ptr + soff, mask=full, other=0.0)
    L = tl.where(full, mv + sv, -float('inf'))
    mx = tl.max(L, axis=1)[:, None]
    ex = tl.exp(L - mx)
    Z = tl.sum(ex, axis=1)
    num = tl.sum(tl.where(ol[None, :] == d, ex, 0.0), axis=1)
    P = num / Z  # (BB,)
    foff = ob[:, None] * (n1 * n2 * D) + c * (n2 * D) + d * D + oe[None, :]
    fblk = tl.load(feat_ptr + foff, mask=mb[:, None] & me[None, :], other=0.0)  # (BB,BD)
    out = tl.sum(P[:, None] * fblk, axis=0)  # (BD,)
    tl.store(o_ptr + a * (n1 * n2 * D) + c * (n2 * D) + d * D + oe, out, mask=me)


class _AttentivePooling(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super(_AttentivePooling, self).__init__()
        self.W_a = nn.Linear(input_dim, input_dim)
        self.W = nn.Linear(input_dim, 1)
        self.act_fn = nn.ReLU()
        self.softmax = nn.functional.softmax


class APNew(nn.Module):
    def __init__(self, out_dim, input_dim):
        super(APNew, self).__init__()
        self.linear = nn.Linear(input_dim, out_dim)
        self.sap_layer = _AttentivePooling(out_dim)
        self.act_fn = nn.ReLU()

    def forward(self, feature_BxTxH, att_mask_BxT):
        x = feature_BxTxH.contiguous()
        n0, n1, n2, n3 = x.shape
        D = self.linear.out_features
        Kin = self.linear.in_features
        M = n0 * n1 * n2

        feat = torch.empty((n0, n1, n2, D), device=x.device, dtype=x.dtype)
        s = torch.empty((M,), device=x.device, dtype=x.dtype)
        BK = triton.next_power_of_2(Kin)
        BD = triton.next_power_of_2(D)
        _feat_s_kernel[(M,)](
            x.view(-1), self.linear.weight, self.linear.bias,
            self.sap_layer.W_a.weight, self.sap_layer.W_a.bias,
            self.sap_layer.W.weight, self.sap_layer.W.bias,
            feat.view(-1), s, M, Kin, D, BK=BK, BD=BD, num_warps=4)

        mask = att_mask_BxT.contiguous()
        out = torch.empty((n0, n1, n2, D), device=x.device, dtype=x.dtype)
        BB = triton.next_power_of_2(n1)
        BL = triton.next_power_of_2(n3)
        _smax_wsum_kernel[(n0 * n1 * n2,)](
            feat.view(-1), mask.view(-1), s, out.view(-1),
            n0, n1, n2, n3, D, BB=BB, BL=BL, BD=BD, num_warps=4)

        return out
