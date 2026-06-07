import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(x_ptr, g_ptr, b_ptr, out_ptr, n_rows, n_cols, eps,
                      BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr):
    pid = tl.program_id(0)
    row_off = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_off = tl.arange(0, BLOCK_COLS)
    rmask = row_off < n_rows
    cmask = col_off < n_cols
    ptrs = x_ptr + row_off[:, None] * n_cols + col_off[None, :]
    mask = rmask[:, None] & cmask[None, :]
    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1)[:, None] / n_cols
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=1)[:, None] / (n_cols - 1)
    std = tl.sqrt(var)
    g = tl.load(g_ptr + col_off, mask=cmask, other=0.0).to(tl.float32)[None, :]
    b = tl.load(b_ptr + col_off, mask=cmask, other=0.0).to(tl.float32)[None, :]
    y = g * xc / (std + eps) + b
    tl.store(out_ptr + row_off[:, None] * n_cols + col_off[None, :], y, mask=mask)


class LayerNormNew(nn.Module):
    def __init__(self, weights, eps=1e-05):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(weights))
        self.beta = nn.Parameter(torch.zeros(weights))
        self.eps = eps

    def forward(self, x):
        n_cols = x.shape[-1]
        x2 = x.contiguous().view(-1, n_cols)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK_COLS = triton.next_power_of_2(n_cols)
        BLOCK_ROWS = 16
        grid = (triton.cdiv(n_rows, BLOCK_ROWS),)
        _layernorm_kernel[grid](x2, self.gamma, self.beta, out, n_rows, n_cols,
                                self.eps, BLOCK_ROWS=BLOCK_ROWS,
                                BLOCK_COLS=BLOCK_COLS, num_warps=1)
        return out.view_as(x)
