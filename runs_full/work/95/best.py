import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _norm_kernel(x_ptr, alpha_ptr, bias_ptr, out_ptr, n_rows, D,
                 eps, ROWS: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    row0 = pid * ROWS
    rows = row0 + tl.arange(0, ROWS)
    rmask = rows < n_rows
    dcol = tl.arange(0, BLOCK_D)
    dmask = dcol < D
    ptrs = x_ptr + rows[:, None] * D + dcol[None, :]
    m = rmask[:, None] & dmask[None, :]
    x = tl.load(ptrs, mask=m, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / D
    xc = tl.where(dmask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / (D - 1)
    std = tl.sqrt(var)
    alpha = tl.load(alpha_ptr + dcol, mask=dmask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + dcol, mask=dmask, other=0.0).to(tl.float32)
    out = alpha[None, :] * xc / (std[:, None] + eps) + bias[None, :]
    tl.store(out_ptr + rows[:, None] * D + dcol[None, :], out, mask=m)


class NormNew(nn.Module):
    def __init__(self, d_model, eps=1e-06):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        D = x.shape[-1]
        x = x.contiguous()
        n_rows = x.numel() // D
        out = torch.empty_like(x)
        BLOCK_D = triton.next_power_of_2(D)
        ROWS = 32
        grid = (triton.cdiv(n_rows, ROWS),)
        _norm_kernel[grid](x, self.alpha, self.bias, out, n_rows, D,
                           self.eps, ROWS=ROWS, BLOCK_D=BLOCK_D, num_warps=4)
        return out
