import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _sum_kernel(x_ptr, out_ptr, rest, stride0, N: tl.constexpr,
                BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < rest
    acc = tl.load(x_ptr + offs, mask=mask, other=0.0)
    for i in tl.static_range(1, N):
        acc += tl.load(x_ptr + i * stride0 + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc, mask=mask)


@triton.jit
def _wsum_kernel(x_ptr, coef_ptr, out_ptr, rest, stride0, N: tl.constexpr,
                 BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < rest
    acc = tl.load(x_ptr + offs, mask=mask, other=0.0)
    for i in tl.static_range(1, N):
        c = tl.load(coef_ptr + i)
        acc += tl.load(x_ptr + i * stride0 + offs, mask=mask, other=0.0) * c
    tl.store(out_ptr + offs, acc, mask=mask)


class SumNew(nn.Module):
    def __init__(self, n, weight=False):
        super(SumNew, self).__init__()
        self.n = n
        self.weight = weight
        self.iter = range(n - 1)
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)

    def forward(self, x):
        x0 = x[0]
        out = torch.empty_like(x0)
        rest = x0.numel()
        stride0 = x.stride(0)
        BLOCK = triton.next_power_of_2(rest) if rest <= 2048 else 2048
        nw = 1 if rest <= 256 else (4 if rest <= 4096 else 8)
        grid = (triton.cdiv(rest, BLOCK),)
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            coef = torch.empty(self.n, device=x.device, dtype=torch.float32)
            coef[1:] = w.to(torch.float32)
            _wsum_kernel[grid](x, coef, out, rest, stride0, self.n,
                               BLOCK=BLOCK, num_warps=nw, num_stages=1)
        else:
            _sum_kernel[grid](x, out, rest, stride0, self.n,
                              BLOCK=BLOCK, num_warps=nw, num_stages=1)
        return out
