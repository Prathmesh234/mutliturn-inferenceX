import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _linmodel_kernel(y_ptr, w_ptr, b_ptr, out_ptr, n_out, length, d_y,
                     WINDOW: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_out
    b = offs // d_y
    j = offs % d_y
    base = b * (length * d_y) + j
    start_t = length - WINDOW
    acc = tl.load(b_ptr) + tl.zeros((BLOCK,), tl.float32)
    for w in range(WINDOW):
        t = start_t + w
        val = tl.load(y_ptr + base + t * d_y, mask=mask, other=0.0)
        wt = tl.load(w_ptr + w)
        acc += val * wt
    tl.store(out_ptr + offs, acc, mask=mask)


class LinearModelNew(nn.Module):
    def __init__(self, context_points: 'int'):
        super().__init__()
        self.window = context_points
        self.linear = nn.Linear(context_points, 1)

    def forward(self, y_c):
        bs, length, d_y = y_c.shape
        y_c = y_c.contiguous()
        out = torch.empty((bs, 1, d_y), device=y_c.device, dtype=y_c.dtype)
        n_out = bs * d_y
        BLOCK = 256
        grid = (triton.cdiv(n_out, BLOCK),)
        _linmodel_kernel[grid](
            y_c, self.linear.weight, self.linear.bias, out,
            n_out, length, d_y, WINDOW=self.window, BLOCK=BLOCK, num_warps=4)
        return out
