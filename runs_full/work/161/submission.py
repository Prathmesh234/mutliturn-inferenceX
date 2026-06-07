import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _clipped_relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    x = tl.maximum(x, 0.0)
    x = tl.minimum(x, 255.0)
    tl.store(out_ptr + offs, x, mask=mask)


class ClippedReLUNew(nn.Module):

    def __init__(self):
        super(ClippedReLUNew, self).__init__()

    def forward(self, x):
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = triton.next_power_of_2(n)
        _clipped_relu_kernel[(1,)](x, out, n,
                                   BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
        return out
