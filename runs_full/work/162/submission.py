import torch
from torch import nn
import numpy as np
import triton
import triton.language as tl


@triton.jit
def _quad_gather_kernel(x_ptr, out_ptr, n, D1, D3, s0, s1, s3, a, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    i = offs // (D1 * D3)
    rem = offs % (D1 * D3)
    j = rem // D3
    l = rem % D3
    x_idx = i * s0 + j * s1 + l * s3
    y = tl.load(x_ptr + x_idx, mask=mask)
    d = y - 1.0
    tl.store(out_ptr + offs, -a * d * d, mask=mask)


@triton.jit
def _rosen_kernel(y_ptr, x_ptr, out_ptr, n_rows, RED, a, b, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    mask = pid < n_rows
    y = tl.load(y_ptr + pid, mask=mask)
    offs = tl.arange(0, BLOCK_SIZE)
    rmask = offs < RED
    base = pid * RED
    vals = tl.load(x_ptr + base + offs, mask=(mask & rmask), other=0.0)
    s = tl.sum(vals, axis=0)
    d = y - 1.0
    tl.store(out_ptr + pid, -a * d * d - b * s, mask=mask)


class RosenbrockNew(nn.Module):

    def __init__(self, n1, n2, a=1.0 / 20.0, b=5.0):
        super(RosenbrockNew, self).__init__()
        self.n1 = n1
        self.n2 = n2
        self.a = a
        self.b = b

    def forward(self, x):
        dim2 = x.ndimension() > 2
        dim1 = x.ndimension() > 1
        if dim2:
            y = x[:, :, 0]
            D0, D1, D3 = y.shape
            s0, s1, s3 = y.stride()
            out = torch.empty((D0, D1, D3), device=x.device, dtype=x.dtype)
            n = D0 * D1 * D3
            BLOCK_SIZE = triton.next_power_of_2(n)
            _quad_gather_kernel[(1,)](x, out, n, D1, D3, s0, s1, s3, self.a,
                                      BLOCK_SIZE=BLOCK_SIZE, num_warps=2)
            result = out
        else:
            xin = x if dim1 else x.unsqueeze(0)
            y = xin[:, 0].contiguous()
            xr = torch.reshape(xin[:, 1:], (xin.size()[0], self.n2, self.n1 - 1))
            xx = xr[:, :, 1:]
            xxx = xr[:, :, 0:-1]
            contrib = ((xx - xxx ** 2) ** 2).reshape(xin.size()[0], -1).contiguous()
            n_rows = xin.size()[0]
            RED = contrib.size()[1]
            out = torch.empty(n_rows, device=x.device, dtype=x.dtype)
            BLOCK_SIZE = triton.next_power_of_2(max(RED, 1))
            _rosen_kernel[(n_rows,)](y, contrib, out, n_rows, RED, self.a, self.b,
                                BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
            result = out
        return result if dim1 else result.squeeze(0)
