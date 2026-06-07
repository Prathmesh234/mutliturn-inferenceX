import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mse_single_kernel(pred_ptr, real_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    p = tl.load(pred_ptr + offs, mask=mask, other=0.0)
    r = tl.load(real_ptr + offs, mask=mask, other=0.0)
    d = r - p
    s = tl.sum(d * d, axis=0)
    tl.store(out_ptr, s / n)


@triton.jit
def _mse_kernel(pred_ptr, real_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    p = tl.load(pred_ptr + offs, mask=mask, other=0.0)
    r = tl.load(real_ptr + offs, mask=mask, other=0.0)
    d = r - p
    block_sum = tl.sum(d * d, axis=0)
    tl.atomic_add(out_ptr, block_sum / n)


class MSENew(nn.Module):
    def __init__(self):
        super(MSENew, self).__init__()

    def forward(self, pred, real):
        pred = pred.contiguous()
        real = real.contiguous()
        n = pred.numel()
        if n <= 16384:
            out = torch.empty((), device=pred.device, dtype=pred.dtype)
            BLOCK_SIZE = triton.next_power_of_2(n)
            nw = 1 if n <= 512 else (2 if n <= 2048 else 4)
            _mse_single_kernel[(1,)](pred, real, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=nw)
            return out
        out = torch.zeros((), device=pred.device, dtype=torch.float32)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _mse_kernel[grid](pred, real, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
        return out.to(pred.dtype)
