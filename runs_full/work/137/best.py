import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr, w_ptr, out_ptr, N, Cin, Cout, H, W, neg_slope,
                  HW: tl.constexpr, K: tl.constexpr, BLOCK_HW: tl.constexpr):
    pid = tl.program_id(0)
    Ctot = Cin + Cout
    n = pid // Ctot
    c = pid % Ctot
    offs = tl.arange(0, BLOCK_HW)
    mask = offs < HW
    out_off = n * Ctot * HW + c * HW + offs
    if c < Cin:
        xval = tl.load(x_ptr + n * Cin * HW + c * HW + offs, mask=mask)
        tl.store(out_ptr + out_off, xval, mask=mask)
    else:
        cout = c - Cin
        oh = offs // W
        ow = offs % W
        acc = tl.zeros((BLOCK_HW,), tl.float32)
        pad = (K - 1) // 2
        for cin in range(Cin):
            for kh in range(K):
                for kw in range(K):
                    ih = oh - pad + kh
                    iw = ow - pad + kw
                    in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & mask
                    x_off = n * Cin * HW + cin * HW + ih * W + iw
                    xval = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)
                    wval = tl.load(w_ptr + cout * Cin * K * K + cin * K * K + kh * K + kw)
                    acc += xval * wval
        acc = tl.where(acc > 0, acc, acc * neg_slope)
        tl.store(out_ptr + out_off, acc, mask=mask)


class make_dense_LReLUNew(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense_LReLUNew, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        x = x.contiguous()
        N, Cin, H, W = x.shape
        Cout = self.conv.out_channels
        K = self.conv.kernel_size[0]
        HW = H * W
        BLOCK_HW = triton.next_power_of_2(HW)
        Ctot = Cin + Cout
        out = torch.empty((N, Ctot, H, W), device=x.device, dtype=x.dtype)
        w = self.conv.weight.contiguous()
        _fused_kernel[(N * Ctot,)](x, w, out, N, Cin, Cout, H, W, 0.01,
                                   HW=HW, K=K, BLOCK_HW=BLOCK_HW, num_warps=4)
        return out
