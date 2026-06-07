import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused(x_ptr, wl_ptr, bl_ptr, wa_ptr, ba_ptr, ww_ptr, bw_ptr, mask_ptr, o_ptr,
           n0, n1, n2, n3, Kin, D,
           BB: tl.constexpr, BL: tl.constexpr, BD: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)  # over (a,c,d)
    a = pid // (n1 * n2)
    r = pid % (n1 * n2)
    c = r // n2
    d = r % n2

    ob = tl.arange(0, BB)
    ol = tl.arange(0, BL)
    oj = tl.arange(0, BD)
    oe = tl.arange(0, BD)
    ok = tl.arange(0, BK)
    mb = ob < n1
    ml = ol < n3
    mj = oj < D
    mk = ok < Kin

    # input[b,l,k]
    ioff = ob[:, None, None] * (n1 * n2 * Kin) + c * (n2 * Kin) + ol[None, :, None] * Kin + ok[None, None, :]
    imask = mb[:, None, None] & ml[None, :, None] & mk[None, None, :]
    inp = tl.load(x_ptr + ioff, mask=imask, other=0.0)  # (BB,BL,BK)

    # feat_full[b,l,j] = sum_k inp*Wl[j,k] + bl[j]
    wl = tl.load(wl_ptr + oj[:, None] * Kin + ok[None, :], mask=mj[:, None] & mk[None, :], other=0.0)  # (D,Kin)
    feat = tl.sum(inp[:, :, None, :] * wl[None, None, :, :], axis=3) + tl.load(bl_ptr + oj, mask=mj, other=0.0)[None, None, :]

    # hidden = relu(Wa @ feat + ba)
    wa = tl.load(wa_ptr + oj[:, None] * D + oj[None, :], mask=mj[:, None] & mj[None, :], other=0.0)  # (D,D)
    hidden = tl.sum(feat[:, :, None, :] * wa[None, None, :, :], axis=3) + tl.load(ba_ptr + oj, mask=mj, other=0.0)[None, None, :]
    hidden = tl.maximum(hidden, 0.0)

    # s[b,l] = ww . hidden + bw
    ww = tl.load(ww_ptr + oj, mask=mj, other=0.0)
    s = tl.sum(hidden * ww[None, None, :], axis=2) + tl.load(bw_ptr)  # (BB,BL)

    # L[b,l] = mask[a,b,c,l] + s
    moff = a * (n1 * n2 * n3) + ob[:, None] * (n2 * n3) + c * n3 + ol[None, :]
    mv = tl.load(mask_ptr + moff, mask=mb[:, None] & ml[None, :], other=0.0)
    L = tl.where(mb[:, None] & ml[None, :], mv + s, -float('inf'))
    mx = tl.max(L, axis=1)[:, None]
    ex = tl.exp(L - mx)
    Z = tl.sum(ex, axis=1)  # (BB,)
    num = tl.sum(tl.where(ol[None, :] == d, ex, 0.0), axis=1)  # (BB,)
    P = num / Z  # (BB,)

    # feat at l==d : (BB,BD)
    fsel = tl.sum(tl.where((ol[None, :, None] == d), feat, 0.0), axis=1)  # (BB,BD)
    out = tl.sum(P[:, None] * fsel, axis=0)  # (BD,)
    tl.store(o_ptr + a * (n1 * n2 * D) + c * (n2 * D) + d * D + oe, out, mask=oe < D)


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
        mask = att_mask_BxT.contiguous()
        n0, n1, n2, n3 = x.shape
        D = self.linear.out_features
        Kin = self.linear.in_features
        out = torch.empty((n0, n1, n2, D), device=x.device, dtype=x.dtype)
        BB = triton.next_power_of_2(n1)
        BL = triton.next_power_of_2(n3)
        BD = triton.next_power_of_2(D)
        BK = triton.next_power_of_2(Kin)
        _fused[(n0 * n1 * n2,)](
            x.view(-1), self.linear.weight, self.linear.bias,
            self.sap_layer.W_a.weight, self.sap_layer.W_a.bias,
            self.sap_layer.W.weight, self.sap_layer.W.bias,
            mask.view(-1), out.view(-1),
            n0, n1, n2, n3, Kin, D, BB=BB, BL=BL, BD=BD, BK=BK, num_warps=4)
        return out
