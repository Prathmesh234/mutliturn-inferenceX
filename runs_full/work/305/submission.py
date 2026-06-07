import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def _cos_env_kernel(d_ptr, out_ptr, n, cutoff, inv_cutoff_pi, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    d = tl.load(d_ptr + offs, mask=mask)
    out = 0.5 * (tl.cos(d * inv_cutoff_pi) + 1.0)
    out = tl.where(d >= cutoff, 0.0, out)
    tl.store(out_ptr + offs, out, mask=mask)

class CosineEnvelopeNew(nn.Module):
    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, d):
        d = d.contiguous()
        out = torch.empty_like(d)
        n = d.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _cos_env_kernel[grid](d, out, n, self.cutoff, math.pi / self.cutoff,
                              BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out
