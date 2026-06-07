import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ln_kernel(x_ptr, out_ptr, scale_ptr, center_ptr, n_rows, N,
               HAS_SCALE: tl.constexpr, HAS_CENTER: tl.constexpr, eps,
               BLOCK_ROWS: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_N)
    rmask = rows < n_rows
    cmask = cols < N
    mask = rmask[:, None] & cmask[None, :]
    x = tl.load(x_ptr + rows[:, None] * N + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / N
    xc = tl.where(cmask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / (N - 1)
    y = xc / (tl.sqrt(var)[:, None] + eps)
    if HAS_SCALE:
        s = tl.load(scale_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
        y = y * s[None, :]
    if HAS_CENTER:
        c = tl.load(center_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
        y = y + c[None, :]
    tl.store(out_ptr + rows[:, None] * N + cols[None, :], y, mask=mask)


class LayerNormNew(nn.Module):
    def __init__(self, features, center=True, scale=False, eps=1e-06):
        super().__init__()
        self.center = center
        self.scale = scale
        self.eps = eps
        self.scale_param = nn.Parameter(torch.ones(features)) if scale else None
        self.center_param = nn.Parameter(torch.zeros(features)) if center else None

    def forward(self, x):
        N = x.shape[-1]
        n_rows = x.numel() // N
        xc = x.contiguous()
        out = torch.empty_like(xc)
        BLOCK_N = triton.next_power_of_2(N)
        BLOCK_ROWS = min(triton.next_power_of_2(n_rows), 256)
        grid = (triton.cdiv(n_rows, BLOCK_ROWS),)
        _ln_kernel[grid](
            xc, out,
            self.scale_param if self.scale else xc,
            self.center_param if self.center else xc,
            n_rows, N, self.scale, self.center, self.eps,
            BLOCK_ROWS=BLOCK_ROWS, BLOCK_N=BLOCK_N, num_warps=4,
        )
        return out.view_as(x)
