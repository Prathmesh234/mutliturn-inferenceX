import torch
import numpy as np
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rbf_kernel(dist_ptr, n_ptr, out_ptr, D, cutoff, R: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < D
    d = tl.load(dist_ptr + offs, mask=mask, other=0.0)
    scale = np.pi / cutoff
    is_zero = d == 0
    is_cut = d >= cutoff
    denom = tl.where(is_zero, 1.0, d)
    for j in tl.static_range(R):
        coef = tl.load(n_ptr + j) * scale
        num = tl.where(is_zero, coef, tl.sin(coef * d))
        out = tl.where(is_cut, 0.0, num / denom)
        tl.store(out_ptr + offs * R + j, out, mask=mask)


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
        D = d.numel()
        n_f = self.n.contiguous()
        BLOCK = 256
        grid = (triton.cdiv(D, BLOCK),)
        _rbf_kernel[grid](d, n_f, out, D, float(self.cutoff),
                          R=R, BLOCK=BLOCK, num_warps=4)
        return out
