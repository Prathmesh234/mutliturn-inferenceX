import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _maxpool_s1_kernel(x_ptr, out_ptr, n_elements, H, W, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    HW = H * W
    j = offs % W
    i = (offs // W) % H
    nc = offs // HW

    i1 = tl.minimum(i + 1, H - 1)
    j1 = tl.minimum(j + 1, W - 1)

    base = nc * HW
    a = tl.load(x_ptr + base + i * W + j, mask=mask)
    b = tl.load(x_ptr + base + i * W + j1, mask=mask)
    c = tl.load(x_ptr + base + i1 * W + j, mask=mask)
    d = tl.load(x_ptr + base + i1 * W + j1, mask=mask)

    m = tl.maximum(tl.maximum(a, b), tl.maximum(c, d))
    tl.store(out_ptr + offs, m, mask=mask)


class MaxPoolStride1New(nn.Module):
    def __init__(self):
        super(MaxPoolStride1New, self).__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        out = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = triton.next_power_of_2(n_elements)
        _maxpool_s1_kernel[(1,)](x, out, n_elements, H, W,
                                 BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
        return out
