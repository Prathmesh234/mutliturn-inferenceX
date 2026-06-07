import torch
from torch import nn as nn
import triton
import triton.language as tl


@triton.jit
def _behler_kernel(cos_ptr, out_ptr, n_elements, coef, zeta, col_pos, col_neg,
                   NCOL: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    c = tl.load(cos_ptr + offs, mask=mask)
    pos = coef * tl.exp(zeta * tl.log(1.0 - c))
    neg = coef * tl.exp(zeta * tl.log(1.0 + c))
    base = offs * NCOL
    tl.store(out_ptr + base + col_pos, pos, mask=mask)
    tl.store(out_ptr + base + col_neg, neg, mask=mask)


class BehlerAngularNew(nn.Module):
    def __init__(self, zetas={1}):
        super(BehlerAngularNew, self).__init__()
        self.zetas = zetas

    def forward(self, cos_theta):
        zetas = list(self.zetas)
        L = len(zetas)
        ncol = 2 * L
        n = cos_theta.numel()
        c = cos_theta.contiguous()
        out = torch.empty((*cos_theta.shape, ncol), dtype=cos_theta.dtype,
                          device=cos_theta.device)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        for i, zeta in enumerate(zetas):
            coef = float(2 ** (1 - zeta))
            _behler_kernel[grid](c, out, n, coef, float(zeta), i, L + i,
                                 NCOL=ncol, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out
