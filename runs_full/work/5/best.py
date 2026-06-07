import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _norm_kernel(x_ptr, g_ptr, b_ptr, out_ptr, M, N, eps,
                 BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    if row >= M:
        return
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    n = N
    mean = tl.sum(x, axis=0) / n
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b
    tl.store(out_ptr + row * N + cols, y, mask=mask)


class NormNew(nn.Module):
    def __init__(self, n_state, axis=-1, epsilon=1e-05):
        super().__init__()
        self.n_state = n_state
        self.g = nn.Parameter(torch.ones([self.n_state]))
        self.b = nn.Parameter(torch.zeros([self.n_state]))
        self.axis = axis
        self.epsilon = epsilon

    def forward(self, x):
        if self.axis != -1 and self.axis != x.dim() - 1:
            u = torch.mean(x, dim=self.axis, keepdim=True)
            s = torch.mean(torch.square(x - u), dim=self.axis, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.epsilon)
            return x * self.g + self.b
        x = x.contiguous()
        N = x.shape[-1]
        M = x.numel() // N
        out = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        grid = (M,)
        _norm_kernel[grid](x, self.g, self.b, out, M, N, self.epsilon,
                           BLOCK_N=BLOCK_N, num_warps=4)
        return out
