import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _se_sum_kernel(a_ptr, b_ptr, out_ptr, N, C, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_out = N * M
    mask = offs < n_out
    n = offs // M
    m = offs % M
    base = n * (C * M) + m
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for c in range(C):
        idx = base + c * M
        a = tl.load(a_ptr + idx, mask=mask, other=0.0)
        b = tl.load(b_ptr + idx, mask=mask, other=0.0)
        d = a - b
        acc += d * d
    tl.store(out_ptr + offs, acc, mask=mask)


class SELossNew(nn.MSELoss):
    def __init__(self):
        super().__init__(reduction='none')

    def forward(self, inputs, target):
        N = inputs.shape[0]
        C = inputs.shape[1]
        M = inputs.numel() // (N * C)
        out_shape = (N,) + tuple(inputs.shape[2:])
        out = torch.empty(out_shape, device=inputs.device, dtype=inputs.dtype)
        n_out = N * M
        BLOCK_SIZE = triton.next_power_of_2(n_out)
        grid = (1,)
        _se_sum_kernel[grid](inputs, target, out, N, C, M, BLOCK_SIZE=BLOCK_SIZE, num_warps=2)
        return out
