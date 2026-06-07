import math
import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _slrelu_kernel(x_ptr, out_ptr, n, slope, scale, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.where(x >= 0, x, x * slope) * scale
    tl.store(out_ptr + offs, y, mask=mask)


class ScaledLeakyReLUNew(nn.Module):

    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        input = input.contiguous()
        out = torch.empty_like(input)
        n = input.numel()
        scale = math.sqrt(2)
        BLOCK_SIZE = triton.next_power_of_2(n)
        grid = (1,)
        _slrelu_kernel[grid](input, out, n, self.negative_slope, scale,
                             BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out
