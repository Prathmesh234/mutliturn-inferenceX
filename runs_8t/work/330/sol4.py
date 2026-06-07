import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(
    x_ptr, wA, wB, wC, wD, bA, bB, bC, bD, out_ptr,
    N, H, W, OH, OW,
    C_in: tl.constexpr, C_out: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C_out * OH * OW
    mask = offs < total

    ow = offs % OW
    t = offs // OW
    oh = t % OH
    t = t // OH
    co = t % C_out
    n = t // C_out

    ph = oh % 2
    i = oh // 2
    pw = ow % 2
    j = ow // 2
    conv_id = ph + 2 * pw
    row_off = ph - 1
    col_off = pw - 1

    bb = tl.where(conv_id == 0, tl.load(bA + co, mask=mask, other=0.0),
         tl.where(conv_id == 1, tl.load(bB + co, mask=mask, other=0.0),
         tl.where(conv_id == 2, tl.load(bC + co, mask=mask, other=0.0),
                  tl.load(bD + co, mask=mask, other=0.0))))
    acc = bb

    for ci in tl.static_range(C_in):
        base = (co * C_in + ci)
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                r = i + kh + row_off
                c = j + kw + col_off
                vld = (r >= 0) & (r < H) & (c >= 0) & (c < W)
                x_idx = ((n * C_in + ci) * H + r) * W + c
                xval = tl.load(x_ptr + x_idx, mask=mask & vld, other=0.0)
                wa = tl.load(wA + (base * 3 + kh) * 3 + kw, mask=mask, other=0.0)
                wb = tl.load(wB + (base * 2 + kh) * 3 + kw, mask=mask & (kh < 2), other=0.0)
                wc = tl.load(wC + (base * 3 + kh) * 2 + kw, mask=mask & (kw < 2), other=0.0)
                wd = tl.load(wD + (base * 2 + kh) * 2 + kw, mask=mask & (kh < 2) & (kw < 2), other=0.0)
                wval = tl.where(conv_id == 0, wa,
                       tl.where(conv_id == 1, wb,
                       tl.where(conv_id == 2, wc, wd)))
                acc += xval * wval

    tl.store(out_ptr + offs, acc, mask=mask)


class UnpoolingAsConvolutionNew(nn.Module):

    def __init__(self, in_kernels, out_kernels):
        super(UnpoolingAsConvolutionNew, self).__init__()
        self.conv_A = nn.Conv2d(in_kernels, out_kernels, kernel_size=(3, 3),
            stride=1, padding=1)
        self.conv_B = nn.Conv2d(in_kernels, out_kernels, kernel_size=(2, 3),
            stride=1, padding=0)
        self.conv_C = nn.Conv2d(in_kernels, out_kernels, kernel_size=(3, 2),
            stride=1, padding=0)
        self.conv_D = nn.Conv2d(in_kernels, out_kernels, kernel_size=(2, 2),
            stride=1, padding=0)

    def forward(self, x):
        x = x.contiguous()
        N, C_in, H, W = x.shape
        C_out = self.conv_A.out_channels
        OH, OW = 2 * H, 2 * W
        out = torch.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)

        total = N * C_out * OH * OW
        BLOCK = 256
        grid = (triton.cdiv(total, BLOCK),)
        _fused_kernel[grid](
            x, self.conv_A.weight, self.conv_B.weight, self.conv_C.weight, self.conv_D.weight,
            self.conv_A.bias, self.conv_B.bias, self.conv_C.bias, self.conv_D.bias, out,
            N, H, W, OH, OW,
            C_in, C_out, BLOCK,
            num_warps=4,
        )
        return out
