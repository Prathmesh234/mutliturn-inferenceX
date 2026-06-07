import torch
from torch import nn as nn
import triton
import triton.language as tl


@triton.jit
def _row_reduce_kernel(x_ptr, out_ptr, M, R, inv_N, DO_MEAN: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr):
    pid = tl.program_id(0)
    row = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.arange(0, BLOCK_R)
    row_mask = row < M
    offs = row[:, None] * R + col[None, :]
    mask = row_mask[:, None] & (col[None, :] < R)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.sum(x, axis=1)
    if DO_MEAN:
        s = s * inv_N
    tl.store(out_ptr + row, s, mask=row_mask)


class AggregateNew(nn.Module):
    def __init__(self, axis, mean=False, keepdim=True):
        super(AggregateNew, self).__init__()
        self.average = mean
        self.axis = axis
        self.keepdim = keepdim

    def forward(self, input, mask=None):
        if mask is not None:
            input = input * mask[..., None]
            y = torch.sum(input, self.axis)
            if self.average:
                N = torch.sum(mask, self.axis, keepdim=self.keepdim)
                N = torch.max(N, other=torch.ones_like(N))
                y = y / N
            return y

        axis = self.axis
        if axis < 0:
            axis += input.dim()
        if axis == input.dim() - 1:
            x = input
        else:
            x = input.movedim(axis, -1).contiguous()
        R = x.shape[-1]
        out_shape = x.shape[:-1]
        M = x.numel() // R
        out = torch.empty(out_shape, device=x.device, dtype=x.dtype)

        BLOCK_M = 128
        BLOCK_R = triton.next_power_of_2(R)
        grid = (triton.cdiv(M, BLOCK_M),)
        inv_N = 1.0 / R if self.average else 1.0
        _row_reduce_kernel[grid](x, out, M, R, inv_N,
                                 DO_MEAN=self.average,
                                 BLOCK_M=BLOCK_M, BLOCK_R=BLOCK_R, num_warps=4)
        return out
