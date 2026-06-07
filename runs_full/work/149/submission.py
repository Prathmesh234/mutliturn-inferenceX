import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _gap_kernel(x_ptr, out_ptr, ROWS, R, BR: tl.constexpr, BC: tl.constexpr):
    row = tl.arange(0, BR)
    col = tl.arange(0, BC)
    mask = (row[:, None] < ROWS) & (col[None, :] < R)
    x = tl.load(x_ptr + row[:, None] * R + col[None, :], mask=mask, other=0.0)
    s = tl.sum(x, axis=1) / R
    tl.store(out_ptr + row, s, mask=row < ROWS)


class GlobalAveragePoolNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        shp = x.shape
        R = shp[-1] * shp[-2]
        rows = x.numel() // R
        out = torch.empty(rows, device=x.device, dtype=x.dtype)
        BR = triton.next_power_of_2(rows)
        BC = triton.next_power_of_2(R)
        _gap_kernel[(1,)](x, out, rows, R, BR=BR, BC=BC, num_warps=1)
        return out.view(*shp[:-2], 1, 1)
