import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv2d_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                   N, CIN, H, W, COUT, HOUT, WOUT,
                   SP, BLOCK: tl.constexpr, CIN_C: tl.constexpr,
                   KH_C: tl.constexpr, KW_C: tl.constexpr):
    nco = tl.program_id(0)
    co = nco % COUT
    n = nco // COUT
    sp = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = sp < SP

    wo = sp % WOUT
    ho = sp // WOUT

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for ci in tl.static_range(CIN_C):
        xbase = (n * CIN + ci) * H
        wbase = (co * CIN + ci) * KH_C
        for kh in tl.static_range(KH_C):
            for kw in tl.static_range(KW_C):
                x_off = (xbase + ho + kh) * W + wo + kw
                xval = tl.load(x_ptr + x_off, mask=mask, other=0.0)
                wval = tl.load(w_ptr + (wbase + kh) * KW_C + kw)
                acc += xval * wval

    acc += tl.load(b_ptr + co)
    tl.store(out_ptr + nco * SP + sp, acc, mask=mask)


class ExampleBackboneNew(nn.Module):
    def __init__(self):
        super(ExampleBackboneNew, self).__init__()
        self.conv = nn.Conv2d(3, 3, 3)

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        x = x.contiguous()
        N, CIN, H, W = x.shape
        COUT, _, KH, KW = self.conv.weight.shape
        HOUT = H - KH + 1
        WOUT = W - KW + 1
        SP = HOUT * WOUT
        out = torch.empty((N, COUT, HOUT, WOUT), device=x.device, dtype=x.dtype)
        BLOCK = 32
        grid = (N * COUT, triton.cdiv(SP, BLOCK))
        _conv2d_kernel[grid](x, self.conv.weight, self.conv.bias, out,
                             N, CIN, H, W, COUT, HOUT, WOUT,
                             SP, BLOCK=BLOCK, CIN_C=CIN, KH_C=KH, KW_C=KW,
                             num_warps=1)
        return [out]
