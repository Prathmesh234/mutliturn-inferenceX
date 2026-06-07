import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rsoftmax_kernel(x_ptr, out_ptr, n_groups, R, G, n_inner,
                     P_BLOCK: tl.constexpr, R_BLOCK: tl.constexpr):
    p = tl.arange(0, P_BLOCK)
    r = tl.arange(0, R_BLOCK)
    mp = p < n_groups
    mr = r < R
    mask = mp[:, None] & mr[None, :]
    i = p % n_inner
    tmp = p // n_inner
    g = tmp % G
    b = tmp // G
    in_base = b * (G * R * n_inner) + g * (R * n_inner) + i
    in_off = in_base[:, None] + r[None, :] * n_inner
    x = tl.load(x_ptr + in_off, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=1)
    e = tl.exp(x - m[:, None])
    s = tl.sum(e, axis=1)
    y = e / s[:, None]
    out_base = b * (R * G * n_inner) + g * n_inner + i
    out_off = out_base[:, None] + r[None, :] * (G * n_inner)
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
            P_BLOCK = triton.next_power_of_2(n_groups)
            R_BLOCK = triton.next_power_of_2(R)
            _rsoftmax_kernel[(1,)](x, out, n_groups, R, G, n_inner,
                                   P_BLOCK=P_BLOCK, R_BLOCK=R_BLOCK, num_warps=4)
        else:
            BLOCK = 1024
            grid = (triton.cdiv(n, BLOCK),)
            _sigmoid_kernel[grid](x, out.view(-1), n, BLOCK=BLOCK, num_warps=4)
        return out


