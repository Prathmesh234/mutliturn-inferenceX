import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gap_kernel(x_ptr, out_ptr, R, HW, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)
    row = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    col = tl.arange(0, BLOCK_C)
    rmask = row < R
    cmask = col < HW
    ptr = x_ptr + row[:, None] * HW + col[None, :]
    x = tl.load(ptr, mask=rmask[:, None] & cmask[None, :], other=0.0)
    s = tl.sum(x, axis=1) / HW
    tl.store(out_ptr + row, s, mask=rmask)


class GlobalAvgPool2dNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        R = N * C
        out = torch.empty((R,), device=x.device, dtype=x.dtype)
        HW = H * W
        BLOCK_C = triton.next_power_of_2(HW)
        BLOCK_R = 4
        grid = (triton.cdiv(R, BLOCK_R),)
        _gap_kernel[grid](x, out, R, HW, BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C, num_warps=2)
        return out.view(N, C)
