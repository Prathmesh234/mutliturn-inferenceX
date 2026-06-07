import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr, w_ptr, out_ptr,
                  N, C, IC: tl.constexpr, OC: tl.constexpr, OUTC, NCH, H, W,
                  KH: tl.constexpr, KW: tl.constexpr, PAD: tl.constexpr,
                  BLOCK: tl.constexpr):
    pid0 = tl.program_id(0)      # n * OUTC + oc
    pid1 = tl.program_id(1)
    n = pid0 // OUTC
    oc = pid0 % OUTC
    HW = H * W
    offs = pid1 * BLOCK + tl.arange(0, BLOCK)
    smask = offs < HW
    oh = offs // W
    ow = offs % W

    in_first = oc < NCH
    in_copy = (oc >= NCH) & (oc < C)
    in_third = oc >= C
    conv_oc = tl.where(in_third, oc - C, oc)
    need_conv = in_first | in_third

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    if need_conv:
        x_base = n * IC * HW
        w_base = conv_oc * IC * KH * KW
        for ic in range(IC):
            xb = x_base + ic * HW
            wb = w_base + ic * KH * KW
            for kh in range(KH):
                ih = oh + kh - PAD
                for kw in range(KW):
                    iw = ow + kw - PAD
                    valid = smask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                    v = tl.load(x_ptr + xb + ih * W + iw, mask=valid, other=0.0)
                    wv = tl.load(w_ptr + wb + kh * KW + kw)
                    acc += v * wv
        acc = tl.maximum(acc, 0.0)

    xv = tl.load(x_ptr + n * C * HW + oc * HW + offs, mask=smask & (oc < C), other=0.0)
    res = tl.where(in_first, xv + acc, tl.where(in_copy, xv, acc))
    tl.store(out_ptr + pid0 * HW + offs, res, mask=smask)


class make_residual_dense_ver1New(nn.Module):

    def __init__(self, nChannels, nChannels_, growthRate, kernel_size=3):
        super(make_residual_dense_ver1New, self).__init__()
        self.conv = nn.Conv2d(nChannels_, growthRate, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)
        self.nChannels_ = nChannels_
        self.nChannels = nChannels
        self.growthrate = growthRate

    def forward(self, x):
        x = x.contiguous()
        N, C, H, W = x.shape
        w = self.conv.weight.contiguous()
        OC, IC, KH, KW = w.shape
        PAD = (KW - 1) // 2
        HW = H * W
        OUTC = C + OC
        NCH = self.nChannels

        out = torch.empty((N, OUTC, H, W), device=x.device, dtype=x.dtype)
        BLOCK = min(triton.next_power_of_2(HW), 1024)
        grid = (N * OUTC, triton.cdiv(HW, BLOCK))
        _fused_kernel[grid](x, w, out, N, C, IC, OC, OUTC, NCH, H, W,
                            KH, KW, PAD, BLOCK=BLOCK, num_warps=4)
        return out
