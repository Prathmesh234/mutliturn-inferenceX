import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mul_kernel(x_ptr, f_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs)
    f = tl.load(f_ptr)
    tl.store(out_ptr + offs, x * f)


class MultiplicationInverseNew(nn.Module):
    def __init__(self, factor=2):
        super(MultiplicationInverseNew, self).__init__()
        self.factor = torch.nn.Parameter(torch.ones(1) * factor)

    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = triton.next_power_of_2(n)
        _mul_kernel[(1,)](x, self.factor, out, BLOCK_SIZE=BLOCK_SIZE, num_warps=2)
        return out

    def inverse(self, y):
        return y / self.factor
