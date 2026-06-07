import torch
import numpy as np
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _fused(x_ptr, y_ptr, out_ptr, G, K, INNER, smooth,
           SQUARE: tl.constexpr, SQV: tl.constexpr,
           BG: tl.constexpr, BK: tl.constexpr, BS: tl.constexpr):
    g = tl.arange(0, BG)[:, None, None]
    k = tl.arange(0, BK)[None, :, None]
    s = tl.arange(0, BS)[None, None, :]
    offs = (g * K + k) * INNER + s
    m = (g < G) & (k < K) & (s < INNER)
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    y = tl.load(y_ptr + offs, mask=m, other=0.0)
    a = x * y
    b = x * (1.0 - y)
    c = (1.0 - x) * y
    if SQUARE:
        a = a * a
        b = b * b
        c = c * c
    tp = tl.sum(a, axis=2)          # (BG, BK)
    fp = tl.sum(b, axis=2)
    fn = tl.sum(c, axis=2)
    vol = tl.sum(y, axis=2)
    v = vol + 1e-6
    if SQV:
        v = v * v
    tp = tp / v
    fp = fp / v
    fn = fn / v
    kk = tl.arange(0, BK)[None, :]
    km = kk < K
    tp = tl.sum(tl.where(km, tp, 0.0), axis=1)   # (BG,)
    fp = tl.sum(tl.where(km, fp, 0.0), axis=1)
    fn = tl.sum(tl.where(km, fn, 0.0), axis=1)
    dc = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    gg = tl.arange(0, BG)
    dc = tl.where(gg < G, dc, 0.0)
    res = tl.sum(dc) / G
    tl.store(out_ptr, -res)


@triton.jit
def _reduce_rows(x_ptr, y_ptr, tpn_ptr, fpn_ptr, fnn_ptr, INNER,
                 SQUARE: tl.constexpr, SQV: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * INNER
    tp = 0.0; fp = 0.0; fn = 0.0; vol = 0.0
    for off in range(0, INNER, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < INNER
        x = tl.load(x_ptr + base + idx, mask=m, other=0.0)
        y = tl.load(y_ptr + base + idx, mask=m, other=0.0)
        a = x * y; b = x * (1.0 - y); c = (1.0 - x) * y
        if SQUARE:
            a = a * a; b = b * b; c = c * c
        tp += tl.sum(a); fp += tl.sum(b); fn += tl.sum(c); vol += tl.sum(y)
    v = vol + 1e-6
    if SQV:
        v = v * v
    tl.store(tpn_ptr + pid, tp / v)
    tl.store(fpn_ptr + pid, fp / v)
    tl.store(fnn_ptr + pid, fn / v)


@triton.jit
def _final(tpn_ptr, fpn_ptr, fnn_ptr, out_ptr, G, K, smooth,
           BB: tl.constexpr, BK: tl.constexpr):
    g = tl.arange(0, BB); k = tl.arange(0, BK)
    offs = g[:, None] * K + k[None, :]
    m = (g[:, None] < G) & (k[None, :] < K)
    tp = tl.sum(tl.load(tpn_ptr + offs, mask=m, other=0.0), axis=1)
    fp = tl.sum(tl.load(fpn_ptr + offs, mask=m, other=0.0), axis=1)
    fn = tl.sum(tl.load(fnn_ptr + offs, mask=m, other=0.0), axis=1)
    dc = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    dc = tl.where(g < G, dc, 0.0)
    res = tl.sum(dc) / G
    tl.store(out_ptr, -res)


def _next_pow2(n):
    return 1 << (max(1, n) - 1).bit_length()


class GDLNew(nn.Module):

    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True,
                 smooth=1.0, square=False, square_volumes=False):
        super(GDLNew, self).__init__()
        self.square_volumes = square_volumes
        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape
        if len(shp_x) != len(shp_y):
            y = y.view((shp_y[0], 1, *shp_y[1:]))
        if all([(i == j) for i, j in zip(x.shape, y.shape)]):
            y_onehot = y
        else:
            gt = y.long()
            y_onehot = torch.zeros(shp_x, device=x.device)
            y_onehot.scatter_(1, gt, 1)
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        if not self.do_bg:
            x = x[:, 1:]
            y_onehot = y_onehot[:, 1:]
        x = x.contiguous().float()
        y_onehot = y_onehot.contiguous().float()
        B = x.shape[0]; C = x.shape[1]
        S = 1
        for d in x.shape[2:]:
            S *= d
        if self.batch_dice:
            xr = x.reshape(B, C, S).permute(1, 0, 2).reshape(C, B * S).contiguous()
            yr = y_onehot.reshape(B, C, S).permute(1, 0, 2).reshape(C, B * S).contiguous()
            rows = C; inner = B * S; G = 1; K = C
        else:
            xr = x.reshape(B * C, S); yr = y_onehot.reshape(B * C, S)
            rows = B * C; inner = S; G = B; K = C

        out = torch.empty((), device=x.device, dtype=torch.float32)
        BG = _next_pow2(G); BK = _next_pow2(K); BS = _next_pow2(inner)
        if BG * BK * BS <= 65536:
            _fused[(1,)](xr, yr, out, G, K, inner, float(self.smooth),
                         SQUARE=self.square, SQV=self.square_volumes,
                         BG=BG, BK=BK, BS=BS, num_warps=4)
        else:
            tpn = torch.empty(rows, device=x.device, dtype=torch.float32)
            fpn = torch.empty(rows, device=x.device, dtype=torch.float32)
            fnn = torch.empty(rows, device=x.device, dtype=torch.float32)
            BLOCK = min(1024, BS)
            _reduce_rows[(rows,)](xr, yr, tpn, fpn, fnn, inner,
                                  SQUARE=self.square, SQV=self.square_volumes,
                                  BLOCK=BLOCK, num_warps=4)
            _final[(1,)](tpn, fpn, fnn, out, G, K, float(self.smooth),
                         BB=BG, BK=BK, num_warps=1)
        return out
