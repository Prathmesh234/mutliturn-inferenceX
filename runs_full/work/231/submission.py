import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    c = 0.7978845608028654
    inner = c * (x + 0.044715 * x * x * x)
    t = 2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0
    out = 0.5 * x * (1.0 + t)
    tl.store(out_ptr + offs, out, mask=mask)

class GELUNew(nn.Module):
    def forward(self, x):
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _gelu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out
