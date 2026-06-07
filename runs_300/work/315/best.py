import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = x * 0.5 * (1.0 + tl.erf(x * 0.7071067811865476))
    tl.store(out_ptr + offs, out, mask=mask)

class GELUNew(nn.Module):
    def forward(self, input):
        x = input.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = triton.next_power_of_2(n)
        _gelu_kernel[(1,)](x, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
        return out
