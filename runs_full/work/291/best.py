import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _clamp_kernel(x_ptr, out_ptr, n, lo, hi, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    x = tl.maximum(x, lo)
    x = tl.minimum(x, hi)
    tl.store(out_ptr + offs, x, mask=mask)


class InvDepthNew(nn.Module):
    def __init__(self, height, width, min_depth=0.5, max_depth=25.0):
        super(InvDepthNew, self).__init__()
        self._min_range = 1.0 / max_depth
        self._max_range = 1.0 / min_depth
        self.w = nn.Parameter(self._init_weights(height, width))
        self._n = height * width
        self._bs = triton.next_power_of_2(height * width)

    def _init_weights(self, height, width):
        r1 = self._min_range
        r2 = self._min_range + (self._max_range - self._min_range) * 0.1
        w_init = (r1 - r2) * torch.rand(1, 1, height, width) + r2
        return w_init

    def forward(self):
        x = self.w
        out = torch.empty_like(x)
        _clamp_kernel[(1,)](x, out, self._n, self._min_range, self._max_range,
                            BLOCK_SIZE=self._bs, num_warps=1)
        return out
