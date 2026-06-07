import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
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

    acc = tl.load(b_ptr + conv_id * C_out + co, mask=mask, other=0.0)

    wbase = (conv_id * C_out + co) * C_in * 9
    for ci in tl.static_range(C_in):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                r = i + kh + row_off
                c = j + kw + col_off
                vld = (r >= 0) & (r < H) & (c >= 0) & (c < W)
                x_idx = ((n * C_in + ci) * H + r) * W + c
                xval = tl.load(x_ptr + x_idx, mask=mask & vld, other=0.0)
                w_idx = wbase + ci * 9 + kh * 3 + kw
                wval = tl.load(w_ptr + w_idx, mask=mask, other=0.0)
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

    def _pack(self, x):
        C_out = self.conv_A.out_channels
        C_in = self.conv_A.in_channels
        w = torch.zeros((4, C_out, C_in, 3, 3), device=x.device, dtype=x.dtype)
        # A:(0,0) full 3x3; B:(1,0) 2x3 -> kh 0..1; C:(0,1) 3x2 -> kw 0..1; D:(1,1) 2x2
        w[0, :, :, :, :] = self.conv_A.weight
        w[1, :, :, 0:2, :] = self.conv_B.weight
        w[2, :, :, :, 0:2] = self.conv_C.weight
        w[3, :, :, 0:2, 0:2] = self.conv_D.weight
        b = torch.stack([self.conv_A.bias, self.conv_B.bias,
                         self.conv_C.bias, self.conv_D.bias], dim=0)
        return w.contiguous(), b.contiguous()

    def forward(self, x):
        x = x.contiguous()
        N, C_in, H, W = x.shape
        C_out = self.conv_A.out_channels
        OH, OW = 2 * H, 2 * W
        out = torch.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)

        w, b = self._pack(x)
        total = N * C_out * OH * OW
        BLOCK = 512
        grid = (triton.cdiv(total, BLOCK),)
        _fused_kernel[grid](
            x, w, b, out,
            N, H, W, OH, OW,
            C_in, C_out, BLOCK,
            num_warps=4,
        )
        return out
