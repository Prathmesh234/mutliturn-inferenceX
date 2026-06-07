import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mul_kernel(x_ptr, f_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    f = tl.load(f_ptr)
    tl.store(out_ptr + offs, x * f, mask=mask)


class MultiplicationInverseNew(nn.Module):
    def __init__(self, factor=2):
        super(MultiplicationInverseNew, self).__init__()
        self.factor = torch.nn.Parameter(torch.ones(1) * factor)

    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = triton.next_power_of_2(n)
        grid = (1,)
        _mul_kernel[grid](x, self.factor, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=2)
        return out

    def inverse(self, y):
        return y / self.factor
