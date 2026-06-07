import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _sine_kernel(x_ptr, out_ptr, n_elements, w0, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.sin(w0 * x), mask=mask)


class SineNew(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _sine_kernel[grid](x, out, n, self.w0, BLOCK_SIZE=BLOCK_SIZE, num_warps=2, num_stages=1)
        return out
