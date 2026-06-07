import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gap_kernel(x_ptr, out_ptr, HW, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * HW
    offs = tl.arange(0, BLOCK)
    mask = offs < HW
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)
    tl.store(out_ptr + pid, s / HW)


class GlobalAvgPool2dNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.contiguous()
        out = torch.empty((N, C), device=x.device, dtype=x.dtype)
        HW = H * W
        BLOCK = triton.next_power_of_2(HW)
        grid = (N * C,)
        _gap_kernel[grid](x, out, HW, BLOCK=BLOCK, num_warps=4)
        return out
