import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _conv_kernel(p0, p1, p2, p3, w_ptr, b_ptr, g_ptr, beta_ptr, out_ptr,
                 NEG_SLOPE, EPS,
                 N, NSEG: tl.constexpr, NCH: tl.constexpr,
                 H: tl.constexpr, W: tl.constexpr,
                 APPLY_ACT: tl.constexpr, APPLY_NORM: tl.constexpr,
                 KSIZE: tl.constexpr, PAD: tl.constexpr,
                 BLOCK_HW: tl.constexpr):
    pid = tl.program_id(0)
    COUT = NCH
    CIN = NSEG * NCH
    n = pid // COUT
    co = pid % COUT
    offs = tl.arange(0, BLOCK_HW)
    HW = H * W
    mask = offs < HW
    h = offs // W
    w = offs % W
    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    seg_base = n * NCH * HW
    w_base = co * CIN * KSIZE * KSIZE
    for s in tl.static_range(NSEG):
        if s == 0:
            ps = p0
        elif s == 1:
            ps = p1
        elif s == 2:
            ps = p2
        else:
            ps = p3
        for lc in tl.static_range(NCH):
            ci = s * NCH + lc
            ch_base = seg_base + lc * HW
            wc = w_base + ci * KSIZE * KSIZE
            for kh in tl.static_range(KSIZE):
                h_in = (h + kh - PAD + H) % H
                for kw in tl.static_range(KSIZE):
                    w_in = (w + kw - PAD + W) % W
                    val = tl.load(ps + ch_base + h_in * W + w_in, mask=mask, other=0.0)
                    wv = tl.load(w_ptr + wc + kh * KSIZE + kw)
                    acc += wv * val
    acc += tl.load(b_ptr + co)
    if APPLY_ACT:
        acc = tl.where(acc >= 0, acc, acc * NEG_SLOPE)
    if APPLY_NORM:
        sm = tl.sum(tl.where(mask, acc, 0.0))
        mean = sm / HW
        d = tl.where(mask, acc - mean, 0.0)
        var = tl.sum(d * d) / HW
        inv = 1.0 / tl.sqrt(var + EPS)
        acc = (acc - mean) * inv * tl.load(g_ptr + co) + tl.load(beta_ptr + co)
    out_base = n * COUT * HW + co * HW
    tl.store(out_ptr + out_base + offs, acc, mask=mask)


class ConvBlockINEDenseNew(nn.Module):

    def __init__(self, n_ch, act='relu', ksize=3, norm='in', padding_mode='circular'):
        super().__init__()
        padding = (ksize - 1) // 2
        if act == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
            self._neg_slope = 0.2
        else:
            self.act = nn.ReLU(True)
            self._neg_slope = 0.0
        self.conv1 = nn.Conv2d(n_ch, n_ch, kernel_size=ksize, padding=padding, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(2 * n_ch, n_ch, kernel_size=ksize, padding=padding, padding_mode=padding_mode)
        self.conv3 = nn.Conv2d(3 * n_ch, n_ch, kernel_size=ksize, padding=padding, padding_mode=padding_mode)
        self.conv4 = nn.Conv2d(4 * n_ch, n_ch, kernel_size=ksize, padding=padding, padding_mode=padding_mode)
        self.norm = norm
        self.ksize = ksize
        self.pad = padding
        self.n_ch = n_ch
        if norm == 'in':
            self.norm1 = nn.InstanceNorm2d(n_ch, affine=True)
            self.norm2 = nn.InstanceNorm2d(n_ch, affine=True)
            self.norm3 = nn.InstanceNorm2d(n_ch, affine=True)

    def _conv(self, segs, conv, apply_act, normmod):
        x0 = segs[0]
        N, NCH, H, W = x0.shape
        NSEG = len(segs)
        COUT = conv.weight.shape[0]
        out = torch.empty((N, COUT, H, W), device=x0.device, dtype=x0.dtype)
        apply_norm = normmod is not None
        if apply_norm:
            g, beta, eps = normmod.weight, normmod.bias, normmod.eps
        else:
            g, beta, eps = conv.bias, conv.bias, 1e-5
        p = list(segs) + [x0] * (4 - NSEG)
        BLOCK_HW = triton.next_power_of_2(H * W)
        grid = (N * COUT,)
        _conv_kernel[grid](
            p[0], p[1], p[2], p[3], conv.weight, conv.bias, g, beta, out,
            self._neg_slope, eps,
            N, NSEG, NCH, H, W,
            apply_act, apply_norm,
            self.ksize, self.pad, BLOCK_HW,
            num_warps=1,
        )
        return out

    def forward(self, x, g=None, b=None):
        x = x.contiguous()
        innorm = self.norm == 'in'
        x1 = self._conv([x], self.conv1, True, self.norm1 if innorm else None)
        x2 = self._conv([x1, x], self.conv2, True, self.norm2 if innorm else None)
        x3 = self._conv([x2, x1, x], self.conv3, True, self.norm3 if innorm else None)
        out = self._conv([x3, x2, x1, x], self.conv4, False, None)
        return out
