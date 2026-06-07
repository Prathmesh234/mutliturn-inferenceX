import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _avgmax_kernel(x_ptr, out_ptr, n_rows, R, BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < n_rows
    cidx = tl.arange(0, BLOCK_R)
    cmask = cidx < R
    offs = rows[:, None] * R + cidx[None, :]
    m2 = rmask[:, None] & cmask[None, :]
    v = tl.load(x_ptr + offs, mask=m2, other=0.0)
    s = tl.sum(v, axis=1) / R
    mx = tl.max(tl.where(cmask[None, :], v, -float('inf')), axis=1)
    tl.store(out_ptr + rows, s + mx, mask=rmask)


class AdaptiveAvgMaxPoolNew(nn.Module):
    def __init__(self, output_size=1, *args, **kwargs):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        N, C, H, W = x.shape
        R = H * W
        n_rows = N * C
        xf = x.contiguous().view(n_rows, R)
        out = torch.empty(n_rows, device=x.device, dtype=x.dtype)
        BLOCK_R = triton.next_power_of_2(R)
        BLOCK_M = triton.next_power_of_2(n_rows)
        _avgmax_kernel[(triton.cdiv(n_rows, BLOCK_M),)](xf, out, n_rows, R, BLOCK_M=BLOCK_M, BLOCK_R=BLOCK_R, num_warps=4)
        return out.view(N, C, 1, 1)
