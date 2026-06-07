import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_kernel(x_ptr, w_ptr, b_ptr, o_ptr, M, Kin, OUT, BK: tl.constexpr):
    pid = tl.program_id(0)
    m = pid // OUT
    j = pid % OUT
    offs = tl.arange(0, BK)
    mask = offs < Kin
    x = tl.load(x_ptr + m * Kin + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + j * Kin + offs, mask=mask, other=0.0)
    acc = tl.sum(x * w) + tl.load(b_ptr + j)
    tl.store(o_ptr + m * OUT + j, acc)


@triton.jit
def _s_kernel(feat_ptr, wa_ptr, ba_ptr, ww_ptr, bw_ptr, s_ptr, M, D, BD: tl.constexpr):
    m = tl.program_id(0)
    offs = tl.arange(0, BD)
    mk = offs < D
    x = tl.load(feat_ptr + m * D + offs, mask=mk, other=0.0)
    oj = offs[:, None]
    ok = offs[None, :]
    wmask = (oj < D) & (ok < D)
    wa = tl.load(wa_ptr + oj * D + ok, mask=wmask, other=0.0)  # (D,D)
    hidden = tl.sum(wa * x[None, :], axis=1) + tl.load(ba_ptr + offs, mask=mk, other=0.0)
    hidden = tl.maximum(hidden, 0.0)
    ww = tl.load(ww_ptr + offs, mask=mk, other=0.0)
    s = tl.sum(ww * hidden) + tl.load(bw_ptr)
    tl.store(s_ptr + m, s)


@triton.jit
def _softmax_kernel(mask_ptr, s_ptr, p_ptr, n0, n1, n2, n3, BL: tl.constexpr):
    pid = tl.program_id(0)  # over (a,b,c) ; rows = n0*n1*n2
    a = pid // (n1 * n2)
    rem = pid % (n1 * n2)
    b = rem // n2
    c = rem % n2
    offs = tl.arange(0, BL)
    ml = offs < n3
    mbase = a * (n1 * n2 * n3) + b * (n2 * n3) + c * n3
    sbase = b * (n1 * n2) + c * n2
    mv = tl.load(mask_ptr + mbase + offs, mask=ml, other=0.0)
    sv = tl.load(s_ptr + sbase + offs, mask=ml, other=0.0)
    L = mv + sv
    L = tl.where(ml, L, -float('inf'))
    mx = tl.max(L)
    e = tl.exp(L - mx)
    z = tl.sum(e)
    p = e / z
    tl.store(p_ptr + mbase + offs, p, mask=ml)


@triton.jit
def _wsum_kernel(feat_ptr, p_ptr, o_ptr, n0, n1, n2, n3, D, BB: tl.constexpr):
    pid = tl.program_id(0)  # over out (a,c,d,e) shape (n0,n1,n2,D)
    a = pid // (n1 * n2 * D)
    r = pid % (n1 * n2 * D)
    c = r // (n2 * D)
    r2 = r % (n2 * D)
    d = r2 // D
    e = r2 % D
    offs = tl.arange(0, BB)
    mb = offs < n1  # b ranges n1 (sum dim)
    pv = tl.load(p_ptr + a * (n1 * n2 * n3) + offs * (n2 * n3) + c * n3 + d, mask=mb, other=0.0)
    fv = tl.load(feat_ptr + offs * (n1 * n2 * D) + c * (n2 * D) + d * D + e, mask=mb, other=0.0)
    out = tl.sum(pv * fv)
    tl.store(o_ptr + a * (n1 * n2 * D) + c * (n2 * D) + d * D + e, out)


class AP(nn.Module):
    def __init__(self, out_dim, input_dim):
        super(AP, self).__init__()
        self.linear = nn.Linear(input_dim, out_dim)
        self.sap_layer = _AttentivePooling(out_dim)
        self.act_fn = nn.ReLU()


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
        BK = triton.next_power_of_2(Kin)
        _linear_kernel[(M * D,)](
            x.view(-1), self.linear.weight, self.linear.bias, feat.view(-1),
            M, Kin, D, BK=BK, num_warps=4)

        s = torch.empty((M,), device=x.device, dtype=x.dtype)
        BD = triton.next_power_of_2(D)
        _s_kernel[(M,)](
            feat.view(-1), self.sap_layer.W_a.weight, self.sap_layer.W_a.bias,
            self.sap_layer.W.weight, self.sap_layer.W.bias, s,
            M, D, BD=BD, num_warps=4)

        mask = att_mask_BxT.contiguous()
        P = torch.empty((n0, n1, n2, n3), device=x.device, dtype=x.dtype)
        BL = triton.next_power_of_2(n3)
        _softmax_kernel[(n0 * n1 * n2,)](
            mask.view(-1), s, P.view(-1), n0, n1, n2, n3, BL=BL, num_warps=4)

        out = torch.empty((n0, n1, n2, D), device=x.device, dtype=x.dtype)
        BB = triton.next_power_of_2(n1)
        _wsum_kernel[(n0 * n1 * n2 * D,)](
            feat.view(-1), P.view(-1), out.view(-1), n0, n1, n2, n3, D, BB=BB, num_warps=4)

        return out
