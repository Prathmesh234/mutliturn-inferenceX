import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _maxpoolpad_kernel(x_ptr, out_ptr, N, C, H, W, Ho, Wo, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * Ho * Wo
    mask = offs < total

    wo = offs % Wo
    t = offs // Wo
    ho = t % Ho
    t = t // Ho
    c = t % C
    n = t // C

    base = (n * C + c) * H
    NEG = float('-inf')
    acc = tl.full((BLOCK,), NEG, tl.float32)

    for kh in tl.static_range(3):
        pr = 2 * ho + 1 + kh
        xr = pr - 1
        for kw in tl.static_range(3):
            pc = 2 * wo + 1 + kw
            xc = pc - 1
            in_x = (pr <= H) & (pc <= W)
            xoff = (base + xr) * W + xc
            v = tl.load(x_ptr + xoff, mask=mask & in_x, other=NEG)
            acc = tl.maximum(acc, v)

    tl.store(out_ptr + offs, acc, mask=mask)


class MaxPoolPadNew(nn.Module):

    def __init__(self):
        super(MaxPoolPadNew, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = x.contiguous()
        N, C, H, W = x.shape
        Ho = H // 2
        Wo = W // 2
        out = torch.empty((N, C, Ho, Wo), device=x.device, dtype=x.dtype)
        total = N * C * Ho * Wo
        BLOCK = 1024
        grid = (triton.cdiv(total, BLOCK),)
        _maxpoolpad_kernel[grid](x, out, N, C, H, W, Ho, Wo, BLOCK=BLOCK, num_warps=2)
        return out
