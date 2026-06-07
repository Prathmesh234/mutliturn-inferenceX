import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    inv_sqrt2 = 0.7071067811865476
    cdf = 0.5 * (1.0 + tl.erf(x * inv_sqrt2))
    out = x * cdf
    tl.store(out_ptr + offs, out, mask=mask)

class GELUNew(nn.Module):
    def forward(self, input):
        x = input.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _gelu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out
