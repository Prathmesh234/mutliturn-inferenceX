import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _mish_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    out = x * (2.0 / (1.0 + tl.exp(-2.0 * sp)) - 1.0)
    tl.store(out_ptr + offs, out, mask=mask)


class MishNew(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = triton.next_power_of_2(n)
        _mish_kernel[(1,)](x, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
        return out
