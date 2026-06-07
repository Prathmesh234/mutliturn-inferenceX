import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, t_ptr, out_ptr,
                  H, W, C: tl.constexpr, HW: tl.constexpr):
    n = tl.program_id(0)
    sp = tl.arange(0, HW)
    oh = sp // W
    ow = sp % W
    cc = tl.arange(0, C)
    base_n = n * C * HW

    # conv1 -> relu, accumulate [C, HW]
    acc = tl.load(b1_ptr + cc)[:, None] + tl.zeros((C, HW), tl.float32)
    for ic in tl.static_range(C):
        in_base = base_n + ic * HW
        for kh in tl.static_range(3):
            ih = oh + kh - 1
            vh = (ih >= 0) & (ih < H)
            for kw in tl.static_range(3):
                iw = ow + kw - 1
                valid = vh & (iw >= 0) & (iw < W)
                xv = tl.load(x_ptr + in_base + ih * W + iw, mask=valid, other=0.0)
                wv = tl.load(w1_ptr + cc * C * 9 + ic * 9 + kh * 3 + kw)
                acc += wv[:, None] * xv[None, :]
    acc = tl.maximum(acc, 0.0)
    # store t [C,HW]
    toff = base_n + cc[:, None] * HW + sp[None, :]
    tl.store(t_ptr + toff, acc)
    tl.debug_barrier()

    # conv2 + residual
    acc2 = tl.load(b2_ptr + cc)[:, None] + tl.zeros((C, HW), tl.float32)
    for ic in tl.static_range(C):
        in_base = base_n + ic * HW
        for kh in tl.static_range(3):
            ih = oh + kh - 1
            vh = (ih >= 0) & (ih < H)
            for kw in tl.static_range(3):
                iw = ow + kw - 1
                valid = vh & (iw >= 0) & (iw < W)
                tv = tl.load(t_ptr + in_base + ih * W + iw, mask=valid, other=0.0)
                wv = tl.load(w2_ptr + cc * C * 9 + ic * 9 + kh * 3 + kw)
                acc2 += wv[:, None] * tv[None, :]
    res = tl.load(x_ptr + toff)
    acc2 += res
    tl.store(out_ptr + toff, acc2)


class _Residual_Block_DBNew(nn.Module):
    def __init__(self, num_ft):
        super(_Residual_Block_DBNew, self).__init__()
        self.conv1 = nn.Conv2d(num_ft, num_ft, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_ft, num_ft, 3, 1, 1, bias=True)

    def forward(self, x):
        x = x.contiguous()
        N, C, H, W = x.shape
        out = torch.empty_like(x)
        t = torch.empty_like(x)
        _fused_kernel[(N,)](
            x, self.conv1.weight, self.conv1.bias,
            self.conv2.weight, self.conv2.bias, t, out,
            H, W, C=C, HW=H * W, num_warps=1,
        )
        return out
