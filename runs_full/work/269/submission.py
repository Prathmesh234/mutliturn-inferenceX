import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _maxpool_kernel(
    x_ptr, out_ptr,
    C, H, W, H1, W1, Hout, Wout,
    sh, sw, dh, dw, pph, ppw, top, left,
    NEG_INF, total,
    BLOCK: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    ow = offs % Wout
    t = offs // Wout
    oh = t % Hout
    t = t // Hout
    c = t % C
    n = t // C
    base = (n * C + c) * H

    acc = tl.full((BLOCK,), NEG_INF, tl.float32)

    for i in tl.static_range(KH):
        for j in tl.static_range(KW):
            p1 = oh * sh + i * dh - pph
            q1 = ow * sw + j * dw - ppw
            in_pad = (p1 >= 0) & (p1 < H1) & (q1 >= 0) & (q1 < W1)
            orig_h = p1 - top
            orig_w = q1 - left
            in_orig = (orig_h >= 0) & (orig_h < H) & (orig_w >= 0) & (orig_w < W)
            ptr = x_ptr + (base + orig_h) * W + orig_w
            v = tl.load(ptr, mask=mask & in_orig, other=NEG_INF)
            v = tl.where(in_orig, v, tl.where(in_pad, 0.0, NEG_INF))
            acc = tl.maximum(acc, v)

    tl.store(out_ptr + offs, acc, mask=mask)


class MaxPool2dStaticSamePaddingNew(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size
        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2
        p = self.pool
        pad = p.padding
        self.pph, self.ppw = (pad, pad) if isinstance(pad, int) else (pad[0], pad[1])
        dil = p.dilation
        self.dh, self.dw = (dil, dil) if isinstance(dil, int) else (dil[0], dil[1])
        self.ceil_mode = p.ceil_mode

    def forward(self, x):
        x = x.contiguous()
        N, C, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        pph, ppw, dh, dw = self.pph, self.ppw, self.dh, self.dw

        extra_h = (-(-W // sw) - 1) * sw - W + kw
        extra_v = (-(-H // sh) - 1) * sh - H + kh
        left = extra_h // 2
        top = extra_v // 2
        H1 = H + extra_v
        W1 = W + extra_h

        if self.ceil_mode:
            Hout = -(-(H1 + 2 * pph - dh * (kh - 1) - 1) // sh) + 1
            Wout = -(-(W1 + 2 * ppw - dw * (kw - 1) - 1) // sw) + 1
        else:
            Hout = (H1 + 2 * pph - dh * (kh - 1) - 1) // sh + 1
            Wout = (W1 + 2 * ppw - dw * (kw - 1) - 1) // sw + 1

        out = torch.empty((N, C, Hout, Wout), device=x.device, dtype=x.dtype)
        total = N * C * Hout * Wout
        if total == 0:
            return out
        BLOCK = triton.next_power_of_2(total)
        _maxpool_kernel[(triton.cdiv(total, BLOCK),)](
            x, out,
            C, H, W, H1, W1, Hout, Wout,
            sh, sw, dh, dw, pph, ppw, top, left,
            float('-inf'), total,
            BLOCK=BLOCK, KH=kh, KW=kw,
            num_warps=1,
        )
        return out
