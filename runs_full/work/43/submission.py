import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _upscale_kernel(x_ptr, out_ptr, n_out, gain,
                    W, OW, factor,
                    BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_out
    ow = offs % OW
    row = offs // OW
    iw = ow // factor
    ih_row = row // factor
    in_idx = ih_row * W + iw
    x = tl.load(x_ptr + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x * gain, mask=mask)


def upscale2d_triton(x, factor=2, gain=1):
    assert x.dim() == 4
    N, C, H, W = x.shape
    if factor == 1 and gain == 1:
        return x
    OH, OW = H * factor, W * factor
    x = x.contiguous()
    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    n_out = out.numel()
    BLOCK_SIZE = 2048
    grid = (triton.cdiv(n_out, BLOCK_SIZE),)
    _upscale_kernel[grid](x, out, n_out, float(gain),
                          W, OW, factor,
                          BLOCK_SIZE=BLOCK_SIZE, num_warps=2)
    return out


class Upscale2dNew(nn.Module):
    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        return upscale2d_triton(x, factor=self.factor, gain=self.gain)
