import torch
from torch import nn
import triton
import triton.language as tl

@triton.jit
def _scalenorm_kernel(x_ptr, g_ptr, out_ptr, n_rows, D, scale, eps,
                      ROWS: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    rmask = rows < n_rows
    offs = tl.arange(0, BLOCK_D)
    cmask = offs < D
    ptrs = x_ptr + rows[:, None] * D + offs[None, :]
    mask = rmask[:, None] & cmask[None, :]
    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    sumsq = tl.sum(x * x, axis=1)
    norm = tl.sqrt(sumsq)
    norm = tl.maximum(norm, eps) * scale
    g = tl.load(g_ptr).to(tl.float32)
    out = x / norm[:, None] * g
    tl.store(out_ptr + rows[:, None] * D + offs[None, :], out, mask=mask)

class ScaleNormNew(nn.Module):
    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.scale = dim ** -0.5
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x):
        D = x.shape[-1]
        n_rows = x.numel() // D
        x = x.contiguous()
        out = torch.empty_like(x)
        BLOCK_D = triton.next_power_of_2(D)
        ROWS = 128
        grid = (triton.cdiv(n_rows, ROWS),)
        _scalenorm_kernel[grid](x, self.g, out, n_rows, D,
                                self.scale, self.eps,
                                ROWS=ROWS, BLOCK_D=BLOCK_D, num_warps=2)
        return out
