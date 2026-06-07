import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _hardswish_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    t = x + 3.0
    t = tl.minimum(tl.maximum(t, 0.0), 6.0)
    out = x * t / 6.0
    tl.store(out_ptr + offs, out, mask=mask)


class HardswishNew(nn.Module):

    @staticmethod
    def forward(x):
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _hardswish_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out
