import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ln_kernel(x_ptr, g_ptr, b_ptr, out_ptr, M, N, eps,
               BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / (N - 1)
    std = tl.sqrt(var)
    g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = g * xc / (std + eps) + b
    tl.store(out_ptr + row * N + cols, y, mask=mask)


class LayerNormNew(nn.Module):
    def __init__(self, weights, eps=1e-05):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(weights))
        self.beta = nn.Parameter(torch.zeros(weights))
        self.eps = eps

    def forward(self, x):
        N = x.shape[-1]
        M = x.numel() // N
        xc = x.contiguous()
        out = torch.empty_like(xc)
        BLOCK_N = triton.next_power_of_2(N)
        _ln_kernel[(M,)](xc, self.gamma, self.beta, out, M, N, self.eps,
                         BLOCK_N=BLOCK_N, num_warps=4)
        return out.view_as(x)
