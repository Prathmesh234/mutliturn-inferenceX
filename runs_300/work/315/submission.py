import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _gelu_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs)
    out = x * 0.5 * (1.0 + tl.erf(x * 0.7071067811865476))
    tl.store(out_ptr + offs, out)

class GELUNew(nn.Module):
    def forward(self, input):
        out = torch.empty_like(input)
        n = input.numel()
        BLOCK_SIZE = triton.next_power_of_2(n)
        _gelu_kernel[(1,)](input, out, BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
        return out
