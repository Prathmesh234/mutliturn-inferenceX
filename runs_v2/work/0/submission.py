import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _sum_dim1_kernel(x_ptr, out_ptr, n_out, D1, inner, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_out
    o = offs // inner
    j = offs % inner
    base = o * (D1 * inner) + j
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for k in range(D1):
        v = tl.load(x_ptr + base + k * inner, mask=mask, other=0.0)
        acc += v
    tl.store(out_ptr + offs, acc, mask=mask)


class SumAggregatorNew(nn.Module):
    def __init__(self):
        super(SumAggregatorNew, self).__init__()

    def forward(self, neighbor):
        x = neighbor.contiguous()
        D0 = x.shape[0]
        D1 = x.shape[1]
        inner = 1
        for s in x.shape[2:]:
            inner *= s
        out_shape = (D0,) + tuple(x.shape[2:])
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        n_out = D0 * inner
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_out, BLOCK_SIZE),)
        _sum_dim1_kernel[grid](x, out, n_out, D1, inner, BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
        return out
