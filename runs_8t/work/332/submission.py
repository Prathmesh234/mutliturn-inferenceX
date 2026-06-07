import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _hloss_kernel(x_ptr, out_ptr, n_out, C, inner, BLOCK: tl.constexpr, CB: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_out
    a = offs // inner
    b = offs % inner
    base = a * C * inner + b
    j = tl.arange(0, CB)
    cmask = j < C
    addr = j[:, None] * inner + base[None, :]
    full = cmask[:, None] & mask[None, :]
    x = tl.load(x_ptr + addr, mask=full, other=0.0).to(tl.float32)
    acc = -tl.sum(tl.exp(x) * x, axis=0)
    tl.store(out_ptr + offs, acc, mask=mask)


class HLossNew(nn.Module):
    def __init__(self):
        super(HLossNew, self).__init__()

    def forward(self, x):
        outer = x.shape[0]
        C = x.shape[1]
        inner = 1
        for s in x.shape[2:]:
            inner *= s
        n_out = outer * inner
        out = torch.empty((outer,) + tuple(x.shape[2:]), device=x.device, dtype=x.dtype)
        x = x.contiguous()
        BLOCK = 256
        CB = triton.next_power_of_2(C)
        grid = (triton.cdiv(n_out, BLOCK),)
        _hloss_kernel[grid](x, out, n_out, C, inner, BLOCK=BLOCK, CB=CB, num_warps=4)
        return out
