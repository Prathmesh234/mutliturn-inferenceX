import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv_interleave_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, H, W, OH, OW,
    row_off, col_off, ph, pw,
    C_in: tl.constexpr, C_out: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C_out * H * W
    mask = offs < total

    j = offs % W
    t = offs // W
    i = t % H
    t = t // H
    co = t % C_out
    n = t // C_out

    acc = tl.load(b_ptr + co, mask=mask, other=0.0)

    for ci in tl.static_range(C_in):
        for kh in tl.static_range(KH):
            for kw in tl.static_range(KW):
                r = i + kh + row_off
                c = j + kw + col_off
                vld = (r >= 0) & (r < H) & (c >= 0) & (c < W)
                x_idx = ((n * C_in + ci) * H + r) * W + c
                xval = tl.load(x_ptr + x_idx, mask=mask & vld, other=0.0)
                w_idx = ((co * C_in + ci) * KH + kh) * KW + kw
                wval = tl.load(w_ptr + w_idx, mask=mask, other=0.0)
                acc += xval * wval

    oh = 2 * i + ph
    ow = 2 * j + pw
    out_idx = ((n * C_out + co) * OH + oh) * OW + ow
    tl.store(out_ptr + out_idx, acc, mask=mask)


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

        total = N * C_out * H * W
        BLOCK = 64
        grid = (triton.cdiv(total, BLOCK),)

        configs = [
            (self.conv_A, 3, 3, -1, -1, 0, 0),
            (self.conv_B, 2, 3, 0, -1, 1, 0),
            (self.conv_C, 3, 2, -1, 0, 0, 1),
            (self.conv_D, 2, 2, 0, 0, 1, 1),
        ]
        for conv, KH, KW, ro, co_off, ph, pw in configs:
            _conv_interleave_kernel[grid](
                x, conv.weight, conv.bias, out,
                N, H, W, OH, OW,
                ro, co_off, ph, pw,
                C_in, C_out, KH, KW, BLOCK,
                num_warps=1,
            )
        return out
