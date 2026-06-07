import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _gap_kernel(x_ptr, out_ptr, R, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < R
    x = tl.load(x_ptr + pid * R + offs, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)
    tl.store(out_ptr + pid, s / R)


class GlobalAveragePoolNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        *lead, H, W = x.shape
        R = H * W
        xc = x.contiguous()
        rows = xc.numel() // R
        out = torch.empty(rows, device=x.device, dtype=x.dtype)
        BLOCK_SIZE = triton.next_power_of_2(R)
        _gap_kernel[(rows,)](xc, out, R, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out.reshape(*lead, 1, 1)
