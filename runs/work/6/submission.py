import torch
from torch import nn as nn
import triton
import triton.language as tl


@triton.jit
def _behler_z1(cos_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    c = tl.load(cos_ptr + offs, mask=mask)
    tl.store(out_ptr + offs * 2, 1.0 - c, mask=mask)
    tl.store(out_ptr + offs * 2 + 1, 1.0 + c, mask=mask)


@triton.jit
def _behler_kernel(cos_ptr, out_ptr, n, s, scale, zeta, C: tl.constexpr,
                   col, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    c = tl.load(cos_ptr + offs, mask=mask)
    base = 1.0 + s * c
    if zeta == 1.0:
        val = scale * base
    else:
        val = scale * tl.exp(zeta * tl.log(base))
    tl.store(out_ptr + offs * C + col, val, mask=mask)


class BehlerAngularNew(nn.Module):
    def __init__(self, zetas={1}):
        super(BehlerAngularNew, self).__init__()
        self.zetas = zetas

    def forward(self, cos_theta):
        cos = cos_theta.contiguous()
        n = cos.numel()
        zlist = list(self.zetas)
        nz = len(zlist)
        C = 2 * nz
        out = torch.empty((*cos.shape, C), dtype=cos.dtype, device=cos.device)
        out_flat = out.view(-1)
        BLOCK = 256 if n <= 4096 else 1024
        nw = 2 if n <= 4096 else 4
        grid = (triton.cdiv(n, BLOCK),)
        if zlist == [1]:
            _behler_z1[grid](cos, out_flat, n, BLOCK=BLOCK, num_warps=nw)
            return out
        col = 0
        for s in (-1.0, 1.0):
            for zeta in zlist:
                scale = 2.0 ** (1 - zeta)
                _behler_kernel[grid](cos, out_flat, n, s, scale, float(zeta),
                                     C, col, BLOCK=BLOCK, num_warps=nw)
                col += 1
        return out
