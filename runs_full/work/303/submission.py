import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _sse_kernel(x_ptr, w_ptr, b_ptr, out_ptr, n_pos, HW, C: tl.constexpr,
                C_BLOCK: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    pos_mask = offs < n_pos
    n = offs // HW
    s = offs % HW
    base = n * (C * HW) + s

    offs_c = tl.arange(0, C_BLOCK)
    c_mask = offs_c < C
    ptrs = base[:, None] + offs_c[None, :] * HW
    tile_mask = pos_mask[:, None] & c_mask[None, :]
    x_tile = tl.load(x_ptr + ptrs, mask=tile_mask, other=0.0)

    for co in range(C):
        w_row = tl.load(w_ptr + co * C + offs_c, mask=c_mask, other=0.0)
        acc = tl.sum(x_tile * w_row[None, :], axis=1)
        bco = tl.load(b_ptr + co)
        y = tl.sigmoid(acc + bco)
        xco = tl.load(x_ptr + base + co * HW, mask=pos_mask, other=0.0)
        tl.store(out_ptr + base + co * HW, xco * y, mask=pos_mask)


class SSENew(nn.Module):
    def __init__(self, in_ch):
        super(SSENew, self).__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1)

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.contiguous()
        out = torch.empty_like(x)
        HW = H * W
        n_pos = N * HW
        w = self.conv.weight.contiguous().view(C, C)
        b = self.conv.bias.contiguous()
        C_BLOCK = triton.next_power_of_2(C)
        BLOCK = 1024
        grid = (triton.cdiv(n_pos, BLOCK),)
        _sse_kernel[grid](x, w, b, out, n_pos, HW, C, C_BLOCK, BLOCK,
                          num_warps=2)
        return out
