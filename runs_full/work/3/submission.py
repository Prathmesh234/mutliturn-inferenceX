import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(x_ptr, g_ptr, b_ptr, out_ptr, n_rows, N, eps,
                      ROW_BLOCK: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    rmask = rows < n_rows
    cols = tl.arange(0, BLOCK_SIZE)
    cmask = cols < N
    ptr = rows[:, None] * N + cols[None, :]
    m = rmask[:, None] & cmask[None, :]
    x = tl.load(x_ptr + ptr, mask=m, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / N
    xc = tl.where(cmask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / (N - 1)
    std = tl.sqrt(var)
    out = g[None, :] * xc / (std[:, None] + eps) + b[None, :]
    tl.store(out_ptr + ptr, out, mask=m)


class LayerNormNew(nn.Module):
    def __init__(self, weights, eps=1e-05):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(weights))
        self.beta = nn.Parameter(torch.zeros(weights))
        self.eps = eps

    def forward(self, x):
        N = x.shape[-1]
        x_c = x.contiguous()
        n_rows = x_c.numel() // N
        out = torch.empty_like(x_c)
        BLOCK_SIZE = triton.next_power_of_2(N)
        ROW_BLOCK = 8
        grid = (triton.cdiv(n_rows, ROW_BLOCK),)
        _layernorm_kernel[grid](x_c, self.gamma, self.beta, out, n_rows, N,
                                self.eps, ROW_BLOCK=ROW_BLOCK,
                                BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
        return out.view_as(x)
