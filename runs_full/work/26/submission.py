import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _meanstd_kernel(x_ptr, out_ptr, M, C, BLOCK_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    n = pid // C
    c = pid % C
    row_start = pid * M
    offs = tl.arange(0, BLOCK_M)
    mask = offs < M
    v = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
    s = tl.sum(v, axis=0)
    sq = tl.sum(v * v, axis=0)
    mean = s / M
    var = sq / M - mean * mean
    out_row = n * (2 * C)
    tl.store(out_ptr + out_row + c, mean)
    tl.store(out_ptr + out_row + C + c, var)


class MeanStdNew(nn.Module):
    def __init__(self):
        super(MeanStdNew, self).__init__()

    def forward(self, x):
        x = x.reshape(x.size(0), x.size(1), -1)
        N, C, M = x.shape
        out = torch.empty((N, 2 * C), device=x.device, dtype=x.dtype)
        BLOCK_M = triton.next_power_of_2(M)
        grid = (N * C,)
        _meanstd_kernel[grid](x, out, M, C, BLOCK_M=BLOCK_M, num_warps=1)
        return out
