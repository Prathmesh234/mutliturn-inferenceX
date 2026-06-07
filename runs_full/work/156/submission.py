import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _hswish_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    r = x + 3.0
    r = tl.minimum(tl.maximum(r, 0.0), 6.0)
    out = x * r * (1.0 / 6.0)
    tl.store(out_ptr + offs, out, mask=mask)


class HSwishNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _hswish_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=2)
        return out
