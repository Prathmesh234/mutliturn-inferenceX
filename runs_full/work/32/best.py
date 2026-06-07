import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ln_kernel(x_ptr, out_ptr, scale_ptr, center_ptr, n_rows, N,
               HAS_SCALE: tl.constexpr, HAS_CENTER: tl.constexpr, eps,
               BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n_rows:
        return
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    row = x_ptr + pid * N
    x = tl.load(row + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / (N - 1)
    std = tl.sqrt(var)
    y = xc / (std + eps)
    if HAS_SCALE:
        s = tl.load(scale_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = y * s
    if HAS_CENTER:
        c = tl.load(center_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = y + c
    tl.store(out_ptr + pid * N + offs, y, mask=mask)


class LayerNormNew(nn.Module):
    def __init__(self, features, center=True, scale=False, eps=1e-06):
        super().__init__()
        self.center = center
        self.scale = scale
        self.eps = eps
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(features))
        else:
            self.scale_param = None
        if self.center:
            self.center_param = nn.Parameter(torch.zeros(features))
        else:
            self.center_param = None

    def forward(self, x):
        N = x.shape[-1]
        n_rows = x.numel() // N
        xc = x.contiguous()
        out = torch.empty_like(xc)
        BLOCK_N = triton.next_power_of_2(N)
        grid = (n_rows,)
        _ln_kernel[grid](
            xc, out,
            self.scale_param if self.scale else xc,
            self.center_param if self.center else xc,
            n_rows, N,
            self.scale, self.center, self.eps,
            BLOCK_N=BLOCK_N, num_warps=4,
        )
        return out.view_as(x)
