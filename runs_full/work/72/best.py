import math
import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _maxpool_kernel(x_ptr, out_ptr, n_elements,
                    N, C, IH, IW, OH, OW,
                    SH, SW, DH, DW, PAD_T, PAD_L,
                    KH: tl.constexpr, KW: tl.constexpr,
                    BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    ow = offs % OW
    t = offs // OW
    oh = t % OH
    t = t // OH
    c = t % C
    n = t // C

    acc = tl.full((BLOCK_SIZE,), float('-inf'), tl.float32)
    base = (n * C + c) * IH
    for ky in tl.static_range(KH):
        ph = oh * SH + ky * DH - PAD_T
        ph_ok = (ph >= 0) & (ph < IH)
        for kx in tl.static_range(KW):
            pw = ow * SW + kx * DW - PAD_L
            pw_ok = (pw >= 0) & (pw < IW)
            ok = ph_ok & pw_ok
            ptr = (base + ph) * IW + pw
            val = tl.load(x_ptr + ptr, mask=mask & ok, other=0.0)
            # out-of-bounds contributes the pad value 0.0
            val = tl.where(ok, val, 0.0)
            acc = tl.maximum(acc, val)

    tl.store(out_ptr + offs, acc, mask=mask)


class MaxPool2dDynamicSamePaddingNew(nn.MaxPool2d):
    def __init__(self, kernel_size, stride, padding=0, dilation=1,
        return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation,
            return_indices, ceil_mode)
        self.stride = [self.stride] * 2 if isinstance(self.stride, int
            ) else self.stride
        self.kernel_size = [self.kernel_size] * 2 if isinstance(self.
            kernel_size, int) else self.kernel_size
        self.dilation = [self.dilation] * 2 if isinstance(self.dilation, int
            ) else self.dilation

    def forward(self, x):
        x = x.contiguous()
        N, C, ih, iw = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        dh, dw = self.dilation
        oh_p, ow_p = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh_p - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
        pad_w = max((ow_p - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
        pad_t = pad_h // 2
        pad_l = pad_w // 2
        ih_pad = ih + pad_h
        iw_pad = iw + pad_w
        OH = (ih_pad - dh * (kh - 1) - 1) // sh + 1
        OW = (iw_pad - dw * (kw - 1) - 1) // sw + 1

        out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
        n_elements = out.numel()
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _maxpool_kernel[grid](x, out, n_elements,
                              N, C, ih, iw, OH, OW,
                              sh, sw, dh, dw, pad_t, pad_l,
                              KH=kh, KW=kw, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
        return out
