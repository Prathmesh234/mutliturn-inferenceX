import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _upsample2_kernel(x_ptr, out_ptr, n_in, W, OW, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_in
    val = tl.load(x_ptr + offs, mask=mask)
    w = offs % W
    row = offs // W
    base = row * 2 * OW + w * 2
    tl.store(out_ptr + base, val, mask=mask)
    tl.store(out_ptr + base + 1, val, mask=mask)
    tl.store(out_ptr + base + OW, val, mask=mask)
    tl.store(out_ptr + base + OW + 1, val, mask=mask)


class MyUpsample2New(nn.Module):

    def forward(self, x):
        x = x.contiguous()
        N, C, H, W = x.shape
        out = torch.empty((N, C, H * 2, W * 2), device=x.device, dtype=x.dtype)
        n_in = x.numel()
        OW = W * 2
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_in, BLOCK_SIZE),)
        _upsample2_kernel[grid](x, out, n_in, W, OW, BLOCK_SIZE=BLOCK_SIZE, num_warps=2)
        return out
