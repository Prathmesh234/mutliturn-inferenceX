import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rsoftmax_kernel(x_ptr, out_ptr, n_groups, R, G, n_inner, R_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n_groups:
        return
    i = pid % n_inner
    tmp = pid // n_inner
    g = tmp % G
    b = tmp // G
    r = tl.arange(0, R_BLOCK)
    mask = r < R
    in_off = b * (G * R * n_inner) + g * (R * n_inner) + i + r * n_inner
    x = tl.load(x_ptr + in_off, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    out_off = b * (R * G * n_inner) + g * n_inner + i + r * (G * n_inner)
    tl.store(out_ptr + out_off, y, mask=mask)


@triton.jit
def _sigmoid_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.sigmoid(x), mask=mask)


class RSoftmaxNew(nn.Module):
    def __init__(self, radix, groups):
        super().__init__()
        self.radix = radix
        self.groups = groups

    def forward(self, x):
        batch = x.size(0)
        x = x.contiguous()
        n = x.numel()
        out = torch.empty_like(x).view(batch, -1)
        if self.radix > 1:
            R = self.radix
            G = self.groups
            n_inner = n // (batch * G * R)
            n_groups = batch * G * n_inner
            R_BLOCK = triton.next_power_of_2(R)
            grid = (n_groups,)
            _rsoftmax_kernel[grid](x, out, n_groups, R, G, n_inner, R_BLOCK=R_BLOCK, num_warps=4)
        else:
            BLOCK = 1024
            grid = (triton.cdiv(n, BLOCK),)
            _sigmoid_kernel[grid](x, out.view(-1), n, BLOCK=BLOCK, num_warps=4)
        return out


