import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr, w_ptr, wf_ptr, lab_ptr, out_ptr,
                  A, B, C, Din, Dout, n_mm, n_out, s, m, eps,
                  BB: tl.constexpr, DD: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mm = offs < n_mm
    row = offs // Dout
    dout = offs % Dout
    BC = B * C
    a = row // BC
    rc = row % BC
    c = rc % C
    acc = tl.zeros([BLOCK], tl.float32)
    for din in range(DD):
        dm = mm & (din < Din)
        ss = tl.zeros([BLOCK], tl.float32)
        for bp in range(BB):
            bm = dm & (bp < B)
            xb = tl.load(x_ptr + ((a * B + bp) * C + c) * Din + din, mask=bm, other=0.0)
            ss += tl.where(bm, xb * xb, 0.0)
        norm = tl.sqrt(ss)
        norm = tl.where(norm < eps, eps, norm)
        xv = tl.load(x_ptr + row * Din + din, mask=dm, other=0.0)
        wv = tl.load(w_ptr + dout * Din + din, mask=dm, other=0.0)
        acc += (xv / norm) * wv
    tl.store(wf_ptr + offs, acc, mask=mm)

    tl.debug_barrier()

    mask = offs < n_out
    DA = Dout * A
    i = offs // DA
    rem = offs % DA
    j = rem // A
    k = rem % A
    lab_k = tl.load(lab_ptr + k, mask=mask, other=0)
    lab_i = tl.load(lab_ptr + i, mask=mask, other=0)
    idxP = ((k * B + lab_k) * C + i) * Dout + j
    valP = tl.load(wf_ptr + idxP, mask=mask, other=0.0)
    P = s * (valP - m)
    S = tl.zeros([BLOCK], tl.float32)
    for b in range(BB):
        bm = mask & (b < B) & (b != lab_i)
        idx = ((i * B + b) * C + j) * Dout + k
        v = tl.load(wf_ptr + idx, mask=bm, other=0.0)
        S += tl.where(bm, tl.exp(s * v), 0.0)
    denom = tl.exp(P) + S
    L = P - tl.log(denom)
    L = tl.where(mask, L, 0.0)
    total = tl.sum(L, axis=0)
    res = -total / n_out
    tl.store(out_ptr, res)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


class AdMSoftmaxLossNew(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        super(AdMSoftmaxLossNew, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        assert len(x) == len(labels)
        x = x.contiguous()
        A, B, C, Din = x.shape
        Dout = self.out_features
        labels = labels.contiguous()
        W = self.fc.weight.contiguous()
        wf = torch.empty((A, B, C, Dout), device=x.device, dtype=x.dtype)
        n_mm = A * B * C * Dout
        n_out = C * Dout * A
        out = torch.empty((), device=x.device, dtype=x.dtype)
        BLOCK = _next_pow2(max(n_mm, n_out))
        _fused_kernel[(1,)](x, W, wf, labels, out, A, B, C, Din, Dout,
                            n_mm, n_out, self.s, self.m, 1e-12,
                            BB=_next_pow2(B), DD=_next_pow2(Din),
                            BLOCK=BLOCK, num_warps=2)
        return out
