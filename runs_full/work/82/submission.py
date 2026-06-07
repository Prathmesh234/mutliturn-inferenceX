import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                  N, Cin, H, W, Cout, OH, OW, PH, PW,
                  KH: tl.constexpr, KW: tl.constexpr, CIN: tl.constexpr,
                  POOL: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * Cout * PH * PW
    mask = offs < total
    pw = offs % PW
    ph = (offs // PW) % PH
    co = (offs // (PW * PH)) % Cout
    n = offs // (PW * PH * Cout)
    bias = tl.load(b_ptr + co, mask=mask, other=0.0)
    base_w = co * Cin * KH * KW
    m = tl.full((BLOCK,), -float('inf'), tl.float32)
    nbase = n * Cin * H * W
    for pi in range(POOL):
        crow = ph * POOL + pi
        for pj in range(POOL):
            ccol = pw * POOL + pj
            base_x = nbase + crow * W + ccol
            acc = bias
            for ci in range(CIN):
                for kh in range(KH):
                    for kw in range(KW):
                        xoff = base_x + ci * H * W + kh * W + kw
                        woff = base_w + ci * KH * KW + kh * KW + kw
                        a = tl.load(x_ptr + xoff, mask=mask, other=0.0)
                        wv = tl.load(w_ptr + woff, mask=mask, other=0.0)
                        acc += a * wv
            m = tl.maximum(m, acc)
    m = tl.maximum(m, 0.0)
    tl.store(out_ptr + offs, m, mask=mask)


class BlockNew(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, pool_size):
        super(BlockNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        N, Cin, H, W = x.shape
        Cout, _, KH, KW = w.shape
        OH = H - KH + 1
        OW = W - KW + 1
        POOL = self.pool.kernel_size if isinstance(self.pool.kernel_size, int) else self.pool.kernel_size[0]
        PH = OH // POOL
        PW = OW // POOL
        out = torch.empty((N, Cout, PH, PW), device=x.device, dtype=torch.float32)
        total = N * Cout * PH * PW
        BLOCK = 64
        grid = (triton.cdiv(total, BLOCK),)
        _fused_kernel[grid](x, w, b, out, N, Cin, H, W, Cout, OH, OW, PH, PW,
                            KH=KH, KW=KW, CIN=Cin, POOL=POOL, BLOCK=BLOCK, num_warps=4)
        return out
