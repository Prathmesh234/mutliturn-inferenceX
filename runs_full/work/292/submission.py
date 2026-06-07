import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv1x1_kernel(x_ptr, out_ptr, w_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    w = tl.load(w_ptr)
    b = tl.load(b_ptr)
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * w + b, mask=mask)


class DummyModelNew(nn.Module):
    def __init__(self, block):
        super(DummyModelNew, self).__init__()
        self.block = block
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = 2048
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _conv1x1_kernel[grid](x, out, self.conv.weight, self.conv.bias, n,
                              BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out
