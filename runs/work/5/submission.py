import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _norm_kernel(x_ptr, g_ptr, b_ptr, out_ptr, n_cols, eps,
                 BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    ptr = x_ptr + row * n_cols + cols
    x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b
    tl.store(out_ptr + row * n_cols + cols, y, mask=mask)


class NormNew(nn.Module):
    def __init__(self, n_state, axis=-1, epsilon=1e-05):
        super().__init__()
        self.n_state = n_state
        self.g = nn.Parameter(torch.ones([self.n_state]))
        self.b = nn.Parameter(torch.zeros([self.n_state]))
        self.axis = axis
        self.epsilon = epsilon

    def forward(self, x):
        assert self.axis == -1 or self.axis == x.ndim - 1
        xc = x.contiguous()
        n_cols = xc.shape[-1]
        n_rows = xc.numel() // n_cols
        out = torch.empty_like(xc)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        _norm_kernel[(n_rows,)](xc, self.g, self.b, out, n_cols,
                                self.epsilon, BLOCK_SIZE=BLOCK_SIZE,
                                num_warps=1)
        return out
