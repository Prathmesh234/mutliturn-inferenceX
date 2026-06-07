import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _sum_dim1_kernel(x_ptr, out_ptr, n_out, reduce_size, inner, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_out
    outer_idx = offs // inner
    inner_idx = offs % inner
    base = outer_idx * reduce_size * inner + inner_idx
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for r in range(reduce_size):
        v = tl.load(x_ptr + base + r * inner, mask=mask, other=0.0)
        acc += v
    tl.store(out_ptr + offs, acc, mask=mask)


class SumAggregatorNew(nn.Module):
    def __init__(self):
        super(SumAggregatorNew, self).__init__()

    def forward(self, neighbor):
        x = neighbor.contiguous()
        shape = x.shape
        outer = shape[0]
        reduce_size = shape[1]
        inner = 1
        for s in shape[2:]:
            inner *= s
        out_shape = (shape[0],) + tuple(shape[2:])
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        n_out = outer * inner
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_out, BLOCK_SIZE),)
        _sum_dim1_kernel[grid](x, out, n_out, reduce_size, inner,
                               BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out
