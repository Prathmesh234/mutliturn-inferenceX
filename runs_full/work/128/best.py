import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _scale_shift_kernel(x_ptr, w_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    w = tl.load(w_ptr)
    b = tl.load(b_ptr)
    tl.store(out_ptr + offs, x * w + b, mask=mask)


class Scale_and_shiftNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = triton.next_power_of_2(n)
        _scale_shift_kernel[(1,)](x, self.weight, self.bias, out, n,
                                  BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out
