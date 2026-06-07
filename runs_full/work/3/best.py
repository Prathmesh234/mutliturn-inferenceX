import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(x_ptr, g_ptr, b_ptr, out_ptr, n_rows, N, eps,
                      BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / (N - 1)
    std = tl.sqrt(var)
    out = g * xc / (std + eps) + b
    tl.store(out_ptr + row * N + offs, out, mask=mask)


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
        grid = (n_rows,)
        _layernorm_kernel[grid](x_c, self.gamma, self.beta, out, n_rows, N,
                                self.eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out.view_as(x)
