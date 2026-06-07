import math
import torch
from torch import nn
from torch.nn import functional as F
import triton
import triton.language as tl


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


@triton.jit
def _lin_kernel(style_ptr, wmod_ptr, bias_ptr, out_ptr, scale, lr_mul,
                in_ch, S, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // in_ch
    ic = pid % in_ch
    s = tl.arange(0, BLOCK)
    mask = s < S
    st = tl.load(style_ptr + b * S + s, mask=mask, other=0.0)
    w = tl.load(wmod_ptr + ic * S + s, mask=mask, other=0.0)
    acc = tl.sum(st * w) * scale + tl.load(bias_ptr + ic) * lr_mul
    tl.store(out_ptr + pid, acc)


@triton.jit
def _weight_kernel(W_ptr, style_ptr, wmod_ptr, bias_ptr, outw_ptr,
                   scale, eps, mod_scale, lr_mul,
                   out_ch, in_ch, INK2, S: tl.constexpr, DEMOD: tl.constexpr,
                   K2: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)  # over batch*out_ch
    b = pid // out_ch
    oc = pid % out_ch
    e = tl.arange(0, BLOCK)
    mask = e < INK2
    icidx = e // K2
    wval = tl.load(W_ptr + oc * INK2 + e, mask=mask, other=0.0)
    # fused modulation: style_out[b, icidx]
    sacc = tl.zeros((BLOCK,), tl.float32)
    for s in range(S):
        st = tl.load(style_ptr + b * S + s)
        wm = tl.load(wmod_ptr + icidx * S + s, mask=mask, other=0.0)
        sacc += st * wm
    sval = sacc * mod_scale + tl.load(bias_ptr + icidx, mask=mask, other=0.0) * lr_mul
    mw = scale * wval * sval
    if DEMOD:
        sq = tl.sum(tl.where(mask, mw * mw, 0.0))
        demod = 1.0 / tl.sqrt(sq + eps)
        mw = mw * demod
    tl.store(outw_ptr + pid * INK2 + e, mw, mask=mask)


@triton.jit
def _fused_kernel(inp_ptr, W_ptr, style_ptr, wmod_ptr, bias_ptr, out_ptr,
                  scale, eps, mod_scale, lr_mul,
                  in_ch, out_ch, H, W,
                  K: tl.constexpr, K2: tl.constexpr, pad: tl.constexpr,
                  S: tl.constexpr, INK2, OH: tl.constexpr, OW: tl.constexpr,
                  OHOW: tl.constexpr, DEMOD: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // out_ch
    oc = pid % out_ch
    e = tl.arange(0, BLOCK)
    mask = e < INK2
    ic = e // K2
    rem = e % K2
    kh = rem // K
    kw = rem % K
    wval = tl.load(W_ptr + oc * INK2 + e, mask=mask, other=0.0)
    sacc = tl.zeros((BLOCK,), tl.float32)
    for s in range(S):
        st = tl.load(style_ptr + b * S + s)
        wm = tl.load(wmod_ptr + ic * S + s, mask=mask, other=0.0)
        sacc += st * wm
    sval = sacc * mod_scale + tl.load(bias_ptr + ic, mask=mask, other=0.0) * lr_mul
    mw = scale * wval * sval
    if DEMOD:
        sq = tl.sum(tl.where(mask, mw * mw, 0.0))
        mw = mw * (1.0 / tl.sqrt(sq + eps))
    in_base = b * in_ch * H * W + ic * H * W
    for p in range(OHOW):
        oh = p // OW
        ow = p % OW
        ih = oh - pad + kh
        iw = ow - pad + kw
        inb = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        x = tl.load(inp_ptr + in_base + ih * W + iw, mask=mask & inb, other=0.0)
        acc = tl.sum(tl.where(mask, x * mw, 0.0))
        tl.store(out_ptr + pid * OHOW + p, acc)


@triton.jit
def _conv_kernel(inp_ptr, w_ptr, out_ptr, total,
                 in_ch: tl.constexpr, out_ch: tl.constexpr,
                 H, W, OH, OW, K: tl.constexpr, pad: tl.constexpr,
                 BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp = tmp // OH
    oc = tmp % out_ch
    b = tmp // out_ch
    acc = tl.zeros((BLOCK,), tl.float32)
    for ic in range(in_ch):
        base_in = (b * in_ch + ic) * H
        base_w = ((b * out_ch + oc) * in_ch + ic) * K
        for kh in range(K):
            ih = oh - pad + kh
            for kw in range(K):
                iw = ow - pad + kw
                inb = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                x = tl.load(inp_ptr + (base_in + ih) * W + iw,
                            mask=mask & inb, other=0.0)
                wv = tl.load(w_ptr + (base_w + kh) * K + kw, mask=mask, other=0.0)
                acc += x * wv
    tl.store(out_ptr + offs, acc, mask=mask)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1,
                 activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.activation = activation
        self.scale = 1 / math.sqrt(in_dim) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            from torch.nn import functional as _F
            out = _F.leaky_relu(out + self.bias * self.lr_mul, 0.2) * (2 ** 0.5)
        else:
            out = F.linear(input, self.weight * self.scale,
                           bias=self.bias * self.lr_mul)
        return out


class ModulatedConv2dNew(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim,
                 demodulate=True, upsample=False, downsample=False,
                 blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.eps = 1e-08
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel,
                                               kernel_size, kernel_size))
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate
        self.style_dim = style_dim

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        oc = self.out_channel
        K = self.kernel_size
        S = self.style_dim

        m = self.modulation
        INK2 = in_channel * K * K
        pad = self.padding
        OH = height + 2 * pad - K + 1
        OW = width + 2 * pad - K + 1
        out = torch.empty((batch * oc, OH, OW), device=input.device,
                          dtype=torch.float32)
        _fused_kernel[(batch * oc,)](
            input.contiguous(), self.weight.contiguous(), style.contiguous(),
            m.weight.contiguous(), m.bias.contiguous(), out,
            self.scale, 1e-08, m.scale, m.lr_mul,
            in_channel, oc, height, width,
            K, K * K, pad, S, INK2, OH, OW, OH * OW, self.demodulate,
            BLOCK=_next_pow2(INK2), num_warps=4)
        return out.view(batch, oc, OH, OW)
