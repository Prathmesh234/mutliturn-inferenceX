import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr, out_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, n,
                  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    w1 = tl.load(w1_ptr)
    b1 = tl.load(b1_ptr)
    w2 = tl.load(w2_ptr)
    b2 = tl.load(b2_ptr)
    h = x * w1 + b1
    h = tl.maximum(h, 0.0)
    y = h * w2 + b2
    tl.store(out_ptr + offs, y, mask=mask)


class SchedulerTestNetNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        w1 = self.conv1.weight.reshape(-1)
        b1 = self.conv1.bias.reshape(-1)
        w2 = self.conv2.weight.reshape(-1)
        b2 = self.conv2.bias.reshape(-1)
        BLOCK_SIZE = 2048
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _fused_kernel[grid](x, out, w1, b1, w2, b2, n,
                            BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out
