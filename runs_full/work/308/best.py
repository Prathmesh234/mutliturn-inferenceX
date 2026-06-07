import torch
import numpy as np
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rbf_kernel(dist_ptr, n_ptr, out_ptr, total, R, cutoff, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    d_idx = offs // R
    j = offs % R
    d = tl.load(dist_ptr + d_idx, mask=mask, other=0.0)
    nval = tl.load(n_ptr + j, mask=mask, other=0.0)
    coef = nval * (np.pi / cutoff)
    denom = tl.where(d == 0, 1.0, d)
    num = tl.where(d == 0, coef, tl.sin(coef * d))
    out = tl.where(d >= cutoff, 0.0, num / denom)
    tl.store(out_ptr + offs, out, mask=mask)


class PainnRadialBasisNew(nn.Module):

    def __init__(self, n_rbf, cutoff, learnable_k):
        super().__init__()
        self.n = torch.arange(1, n_rbf + 1).float()
        if learnable_k:
            self.n = nn.Parameter(self.n)
        self.cutoff = cutoff

    def forward(self, dist):
        d = dist.contiguous()
        R = self.n.numel()
        out = torch.empty(d.shape + (R,), device=d.device, dtype=torch.float32)
        total = out.numel()
        n_f = self.n.contiguous()
        BLOCK = 1024
        grid = (triton.cdiv(total, BLOCK),)
        _rbf_kernel[grid](d, n_f, out, total, R, float(self.cutoff),
                          BLOCK=BLOCK, num_warps=4)
        return out
