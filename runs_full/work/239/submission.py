import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _rsoftmax_kernel(x_ptr, out_ptr, B, C, R, D, n_groups,
                     R_POW2: tl.constexpr, BLOCK_G: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs_g = pid * BLOCK_G + tl.arange(0, BLOCK_G)
    gmask = offs_g < n_groups
    d = offs_g % D
    tmp = offs_g // D
    c = tmp % C
    b = tmp // C

    offs_r = tl.arange(0, R_POW2)
    rmask = offs_r < R
    # 2D: [BLOCK_G, R_POW2]
    in_idx = (b * (C * R * D) + c * (R * D) + d)[:, None] + (offs_r * D)[None, :]
    mask = gmask[:, None] & rmask[None, :]
    vals = tl.load(x_ptr + in_idx, mask=mask, other=-float('inf'))
    m = tl.max(vals, axis=1)
    e = tl.exp(vals - m[:, None])
    s = tl.sum(e, axis=1)
    res = e / s[:, None]
    out_idx = (b * (R * C * D) + c * D + d)[:, None] + (offs_r * (C * D))[None, :]
    tl.store(out_ptr + out_idx, res, mask=mask)


@triton.jit
def _sigmoid_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, 1.0 / (1.0 + tl.exp(-x)), mask=mask)


class rSoftMaxNew(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        x = x.contiguous()
        if self.radix > 1:
            B = batch
            C = self.cardinality
            R = self.radix
            D = x.numel() // (B * C * R)
            out = torch.empty(B * C * R * D, device=x.device, dtype=x.dtype)
            n_groups = B * C * D
            R_POW2 = triton.next_power_of_2(R)
            BLOCK_G = 256
            grid = (triton.cdiv(n_groups, BLOCK_G),)
            _rsoftmax_kernel[grid](x, out, B, C, R, D, n_groups,
                                   R_POW2=R_POW2, BLOCK_G=BLOCK_G, num_warps=1)
            return out.view(batch, -1)
        else:
            out = torch.empty_like(x)
            n = x.numel()
            BLOCK = 1024
            grid = (triton.cdiv(n, BLOCK),)
            _sigmoid_kernel[grid](x, out, n, BLOCK=BLOCK, num_warps=4)
            return out.view(batch, -1)
