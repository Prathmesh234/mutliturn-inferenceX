import math
import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW, OC, OH, OW,
    SH, SW, DH, DW, pad_top, pad_left,
    GROUPS,
    HAS_BIAS: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr, ICPG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * OC * OH * OW
    mask = offs < total

    ow = offs % OW
    t = offs // OW
    oh = t % OH
    t = t // OH
    oc = t % OC
    n = t // OC

    ocpg = OC // GROUPS
    g = oc // ocpg
    ic_start = g * ICPG

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for ic in range(ICPG):
        for kh in range(KH):
            for kw in range(KW):
                ih = oh * SH - pad_top + kh * DH
                iw = ow * SW - pad_left + kw * DW
                valid = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
                x_idx = ((n * IC + ic_start + ic) * IH + ih) * IW + iw
                xv = tl.load(x_ptr + x_idx, mask=mask & valid, other=0.0)
                w_idx = ((oc * ICPG + ic) * KH + kh) * KW + kw
                wv = tl.load(w_ptr + w_idx, mask=mask, other=0.0)
                acc += xv * wv

    if HAS_BIAS:
        acc += tl.load(b_ptr + oc, mask=mask, other=0.0)

    tl.store(out_ptr + offs, acc, mask=mask)


class Conv2dDynamicSamePaddingNew(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0,
            dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]
            ] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        dh, dw = self.dilation
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
        pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
        pad_top = pad_h // 2
        pad_left = pad_w // 2

        x = x.contiguous()
        N = x.size(0)
        IC = x.size(1)
        OC = self.weight.size(0)
        ICPG = self.weight.size(1)

        out = torch.empty((N, OC, oh, ow), device=x.device, dtype=x.dtype)
        total = N * OC * oh * ow
        BLOCK = 256
        grid = (triton.cdiv(total, BLOCK),)
        _conv2d_kernel[grid](
            x, self.weight, self.bias if self.bias is not None else x, out,
            N, IC, ih, iw, OC, oh, ow,
            sh, sw, dh, dw, pad_top, pad_left,
            self.groups,
            self.bias is not None,
            kh, kw, ICPG,
            BLOCK=BLOCK, num_warps=1,
        )
        return out
