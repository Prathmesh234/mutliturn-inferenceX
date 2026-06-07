import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(x_ptr, w_ptr, b_ptr, out_ptr, M, N, eps,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    rmask = rows < M
    cmask = cols < N
    offs = rows[:, None] * N + cols[None, :]
    mask = rmask[:, None] & cmask[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    n = N
    mean = tl.sum(x, axis=1) / n
    xc = tl.where(mask, x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / (n - 1)
    std = tl.sqrt(var)
    w = tl.load(w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
    y = w[None, :] * xc / (std[:, None] + eps) + b[None, :]
    tl.store(out_ptr + offs, y, mask=mask)


class LayerNormNew(nn.Module):
    def __init__(self, features, eps=1e-06):
        super(LayerNormNew, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        N = x.shape[-1]
        M = x.numel() // N
        xc = x.contiguous()
        out = torch.empty_like(xc)
        BLOCK_N = triton.next_power_of_2(N)
        BLOCK_M = 128
        grid = (triton.cdiv(M, BLOCK_M),)
        _layernorm_kernel[grid](xc, self.a_2, self.b_2, out, M, N, self.eps,
                                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=2)
        return out.view_as(x)
