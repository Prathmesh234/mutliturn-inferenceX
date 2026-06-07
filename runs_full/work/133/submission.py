import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mish_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    e = tl.exp(x)
    n = e * e + 2.0 * e
    res = tl.where(x > 20.0, x, x * n / (n + 2.0))
    tl.store(out_ptr + offs, res, mask=mask)


class MishNew(nn.Module):
    def forward(self, x):
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = triton.next_power_of_2(n)
        grid = (1,)
        _mish_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
        return out
