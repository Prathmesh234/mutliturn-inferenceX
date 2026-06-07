import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv1x1_kernel(x_ptr, w_ptr, b_ptr, out_ptr, M, HW,
                    C_in: tl.constexpr, C_out: tl.constexpr,
                    BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M
    n = offs // HW
    s = offs % HW

    base_in = n * (C_in * HW) + s
    base_out = n * (C_out * HW) + s
    for co in range(C_out):
        acc = tl.zeros((BLOCK_M,), tl.float32) + tl.load(b_ptr + co)
        for ci in range(C_in):
            xci = tl.load(x_ptr + base_in + ci * HW, mask=mask, other=0.0)
            acc += xci * tl.load(w_ptr + co * C_in + ci)
        tl.store(out_ptr + base_out + co * HW, acc, mask=mask)


class OutConvNew(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConvNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        N, C_in, H, W = x.shape
        C_out = self.conv.out_channels
        HW = H * W
        M = N * HW
        x = x.contiguous()
        out = torch.empty((N, C_out, H, W), device=x.device, dtype=x.dtype)
        w = self.conv.weight.view(C_out, C_in).contiguous()
        b = self.conv.bias.contiguous()
        BLOCK_M = 1024
        grid = (triton.cdiv(M, BLOCK_M),)
        _conv1x1_kernel[grid](x, w, b, out, M, HW, C_in, C_out,
                              BLOCK_M=BLOCK_M, num_warps=1)
        return out
