import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv_relu_kernel(x_ptr, w_ptr, out_ptr,
                      N, IC: tl.constexpr, OC: tl.constexpr, H, W,
                      KH: tl.constexpr, KW: tl.constexpr, PAD: tl.constexpr,
                      BLOCK: tl.constexpr):
    pid0 = tl.program_id(0)      # n * OC + oc
    pid1 = tl.program_id(1)      # spatial block
    n = pid0 // OC
    oc = pid0 % OC
    HW = H * W
    offs = pid1 * BLOCK + tl.arange(0, BLOCK)
    smask = offs < HW
    oh = offs // W
    ow = offs % W

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    x_base = n * IC * HW
    w_base = oc * IC * KH * KW
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
    tl.store(out_ptr + pid0 * HW + offs, acc, mask=smask)


@triton.jit
def _assemble_kernel(x_ptr, ool_ptr, out_ptr,
                     N, C, GR, OUTC, NCH, HW, total,
                     BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    chw = C * HW
    ochw = OUTC * HW
    grhw = GR * HW

    n = offs // ochw
    rem = offs % ochw
    oc = rem // HW
    hw = rem % HW

    # region 1: oc < NCH  -> x[n,oc] + ool[n,oc]
    # region 2: NCH <= oc < C -> x[n,oc]
    # region 3: oc >= C -> ool[n, oc-C]
    x_idx = n * chw + oc * HW + hw
    ool_idx = n * grhw + oc * HW + hw
    ool_idx3 = n * grhw + (oc - C) * HW + hw

    in_first = oc < NCH
    in_copy = (oc >= NCH) & (oc < C)
    in_third = oc >= C

    xv = tl.load(x_ptr + x_idx, mask=mask & (oc < C), other=0.0)
    oolv = tl.load(ool_ptr + ool_idx, mask=mask & in_first, other=0.0)
    oolv3 = tl.load(ool_ptr + ool_idx3, mask=mask & in_third, other=0.0)

    res = tl.where(in_first, xv + oolv,
                   tl.where(in_copy, xv, oolv3))
    tl.store(out_ptr + offs, res, mask=mask)


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

        ool = torch.empty((N, OC, H, W), device=x.device, dtype=x.dtype)
        BLOCK = min(triton.next_power_of_2(HW), 1024)
        grid = (N * OC, triton.cdiv(HW, BLOCK))
        _conv_relu_kernel[grid](x, w, ool, N, IC, OC, H, W, KH, KW, PAD,
                                BLOCK=BLOCK, num_warps=4)

        NCH = self.nChannels
        GR = OC
        OUTC = C + GR
        out = torch.empty((N, OUTC, H, W), device=x.device, dtype=x.dtype)
        total = N * OUTC * HW
        B2 = 1024
        grid2 = (triton.cdiv(total, B2),)
        _assemble_kernel[grid2](x, ool, out, N, C, GR, OUTC, NCH, HW, total,
                                BLOCK=B2, num_warps=4)
        return out
