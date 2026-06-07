import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _sum_dim1_kernel(x_ptr, out_ptr, n_out, inner, REDUCE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_out
    oi = offs // inner
    ri = offs % inner
    base = oi * (REDUCE * inner) + ri
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for j in tl.static_range(REDUCE):
        v = tl.load(x_ptr + base + j * inner, mask=mask, other=0.0)
        acc += v
    tl.store(out_ptr + offs, acc, mask=mask)


class SumAggregatorNew(nn.Module):

    def __init__(self):
        super(SumAggregatorNew, self).__init__()

    def forward(self, neighbor):
        x = neighbor.contiguous()
        shape = x.shape
        outer = shape[0]
        reduce = shape[1]
        inner = 1
        for d in shape[2:]:
            inner *= d
        n_out = outer * inner
        out = torch.empty((outer,) + tuple(shape[2:]), device=x.device, dtype=x.dtype)
        BLOCK_SIZE = triton.next_power_of_2(n_out)
        grid = (triton.cdiv(n_out, BLOCK_SIZE),)
        _sum_dim1_kernel[grid](x, out, n_out, inner, reduce, BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
        return out
