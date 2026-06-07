import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _norm_kernel(x_ptr, scale_ptr, out_ptr, n_spatial, C, HW, eps,
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_spatial
    n = offs // HW
    inner = offs % HW
    base = n * C * HW + inner

    sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for c in range(C):
        x = tl.load(x_ptr + base + c * HW, mask=mask, other=0.0).to(tl.float32)
        sq += x * x
    norm = tl.sqrt(sq) + eps

    for c in range(C):
        x = tl.load(x_ptr + base + c * HW, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(scale_ptr + c).to(tl.float32)
        tl.store(out_ptr + base + c * HW, x / norm * s, mask=mask)


class CaffeNormalizeNew(nn.Module):
    def __init__(self, features, eps=1e-07):
        super(CaffeNormalizeNew, self).__init__()
        self.scale = nn.Parameter(10.0 * torch.ones(features))
        self.eps = eps

    def forward(self, x):
        x = x.contiguous()
        N, C, H, W = x.shape
        HW = H * W
        n_spatial = N * HW
        out = torch.empty_like(x)
        BLOCK_SIZE = 128
        grid = (triton.cdiv(n_spatial, BLOCK_SIZE),)
        _norm_kernel[grid](x, self.scale, out, n_spatial, C, HW, self.eps,
                           BLOCK_SIZE=BLOCK_SIZE, num_warps=2)
        return out
