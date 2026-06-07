import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _channel_pool_kernel(x_ptr, out_ptr, n_spatial, C, HW, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_spatial
    b = offs // HW
    hw = offs % HW
    base = b * C * HW + hw

    maxv = tl.full((BLOCK_SIZE,), -float('inf'), tl.float32)
    acc = tl.zeros((BLOCK_SIZE,), tl.float32)
    for c in range(C):
        v = tl.load(x_ptr + base + c * HW, mask=mask, other=-float('inf'))
        maxv = tl.maximum(maxv, v)
        acc += tl.where(mask, v, 0.0)
    meanv = acc / C

    out_base = b * 2 * HW + hw
    tl.store(out_ptr + out_base, maxv, mask=mask)
    tl.store(out_ptr + out_base + HW, meanv, mask=mask)


class ChannelPoolNew(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        HW = H * W
        out = torch.empty((B, 2, H, W), device=x.device, dtype=x.dtype)
        n_spatial = B * HW
        BLOCK_SIZE = 128
        grid = (triton.cdiv(n_spatial, BLOCK_SIZE),)
        _channel_pool_kernel[grid](x, out, n_spatial, C, HW, BLOCK_SIZE=BLOCK_SIZE, num_warps=2)
        return out
