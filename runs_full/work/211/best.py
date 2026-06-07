import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


class SkipConnectionNew(nn.Module):
    """Linearize gradients, to make learning easier."""

    def __init__(self, *fn):
        super().__init__()
        self.fn = nn.Sequential(*fn)

    def forward(self, x):
        y = self.fn(x)
        if x.shape[-1] < y.shape[-1]:
            return y
        # else: x + y (full) or x[..., :y.shape[-1]] + y (sliced)
        if x.shape == y.shape:
            xc = x.contiguous()
        else:
            xc = x[..., :y.shape[-1]].contiguous()
        yc = y.contiguous()
        out = torch.empty_like(yc)
        n = yc.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _add_kernel[grid](xc, yc, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out
