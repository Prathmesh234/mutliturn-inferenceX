import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _maxpool_kernel(
    x_ptr, out_ptr,
    N, C, H, W, H1, W1, Hout, Wout,
    sh, sw, kh, kw, dh, dw, pph, ppw, top, left,
    NEG_INF,
    BLOCK: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * Hout * Wout
    mask = offs < total

    ow = offs % Wout
    t = offs // Wout
    oh = t % Hout
    t = t // Hout
    c = t % C
    n = t // C

    acc = tl.full((BLOCK,), NEG_INF, tl.float32)

    for i in tl.static_range(KH):
        for j in tl.static_range(KW):
            p1 = oh * sh + i * dh - pph
            q1 = ow * sw + j * dw - ppw
            in_pad = (p1 >= 0) & (p1 < H1) & (q1 >= 0) & (q1 < W1)
            orig_h = p1 - top
            orig_w = q1 - left
            in_orig = (orig_h >= 0) & (orig_h < H) & (orig_w >= 0) & (orig_w < W)
            ptr = x_ptr + ((n * C + c) * H + orig_h) * W + orig_w
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

    def forward(self, x):
        x = x.contiguous()
        N, C, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride

        p = self.pool
        pad = p.padding
        if isinstance(pad, int):
            pph, ppw = pad, pad
        else:
            pph, ppw = pad[0], pad[1]
        dil = p.dilation
        if isinstance(dil, int):
            dh, dw = dil, dil
        else:
            dh, dw = dil[0], dil[1]
        ceil_mode = p.ceil_mode

        extra_h = (math.ceil(W / sw) - 1) * sw - W + kw
        extra_v = (math.ceil(H / sh) - 1) * sh - H + kh
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        H1 = H + top + bottom
        W1 = W + left + right

        def out_dim(L, pp, k, d, s):
            num = L + 2 * pp - d * (k - 1) - 1
            if ceil_mode:
                return num // s + 1 + (1 if num % s != 0 else 0)
            return num // s + 1

        Hout = out_dim(H1, pph, kh, dh, sh)
        Wout = out_dim(W1, ppw, kw, dw, sw)

        out = torch.empty((N, C, Hout, Wout), device=x.device, dtype=x.dtype)
        total = N * C * Hout * Wout
        if total == 0:
            return out
        BLOCK = 256
        grid = (triton.cdiv(total, BLOCK),)
        NEG_INF = float('-inf')
        _maxpool_kernel[grid](
            x, out,
            N, C, H, W, H1, W1, Hout, Wout,
            sh, sw, kh, kw, dh, dw, pph, ppw, top, left,
            NEG_INF,
            BLOCK=BLOCK, KH=kh, KW=kw,
            num_warps=4,
        )
        return out
