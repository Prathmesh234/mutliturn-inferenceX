import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ln_kernel(x_ptr, w_ptr, b_ptr, out_ptr, n_rows, N, eps,
               BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_N)
    rmask = rows < n_rows
    cmask = cols < N
    ptrs = rows[:, None] * N + cols[None, :]
    mask = rmask[:, None] & cmask[None, :]
    x = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / N
    xc = tl.where(cmask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
    y = xc * rstd[:, None] * w[None, :] + b[None, :]
    tl.store(out_ptr + ptrs, y, mask=mask)


class BertLayerNormNew(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        N = x.shape[-1]
        xf = x.contiguous()
        n_rows = xf.numel() // N
        out = torch.empty_like(xf)
        BLOCK_N = triton.next_power_of_2(N)
        BLOCK_R = 8
        grid = (triton.cdiv(n_rows, BLOCK_R),)
        _ln_kernel[grid](xf, self.weight, self.bias, out, n_rows, N,
                         self.variance_epsilon, BLOCK_R=BLOCK_R, BLOCK_N=BLOCK_N, num_warps=2)
        return out.view_as(x)
