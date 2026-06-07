import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _two_conv_relu_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, tmp_ptr, out_ptr,
                          N, Cin, Cmid, Cout, H, W,
                          HW: tl.constexpr, W_c: tl.constexpr,
                          CIN: tl.constexpr, CMID: tl.constexpr,
                          BLOCK1: tl.constexpr, BLOCK2: tl.constexpr):
    n = tl.program_id(0)

    # ---- Layer 1: produce tmp[n, Cmid, H, W] ----
    offs1 = tl.arange(0, BLOCK1)          # over Cmid*HW
    m1 = offs1 < (Cmid * HW)
    co1 = offs1 // HW
    hw1 = offs1 % HW
    h1 = hw1 // W_c
    w1 = hw1 % W_c

    acc1 = tl.zeros((BLOCK1,), dtype=tl.float32)
    xb = n * Cin * HW
    for ci in tl.static_range(CIN):
        xc = xb + ci * HW
        wc = ci * 9
        for kh in tl.static_range(3):
            ih = h1 + kh - 1
            vh = (ih >= 0) & (ih < H)
            for kw in tl.static_range(3):
                iw = w1 + kw - 1
                valid = vh & (iw >= 0) & (iw < W) & m1
                inp = tl.load(x_ptr + xc + ih * W_c + iw, mask=valid, other=0.0)
                wv = tl.load(w1_ptr + co1 * CIN * 9 + wc + kh * 3 + kw, mask=m1, other=0.0)
                acc1 += inp * wv
    acc1 += tl.load(b1_ptr + co1, mask=m1, other=0.0)
    acc1 = tl.maximum(acc1, 0.0)
    tb = n * Cmid * HW
    tl.store(tmp_ptr + tb + offs1, acc1, mask=m1)

    # ---- Layer 2: read tmp -> out[n, Cout, H, W] ----
    offs2 = tl.arange(0, BLOCK2)
    m2 = offs2 < (Cout * HW)
    co2 = offs2 // HW
    hw2 = offs2 % HW
    h2 = hw2 // W_c
    w2 = hw2 % W_c

    acc2 = tl.zeros((BLOCK2,), dtype=tl.float32)
    for ci in tl.static_range(CMID):
        tc = tb + ci * HW
        wc = ci * 9
        for kh in tl.static_range(3):
            ih = h2 + kh - 1
            vh = (ih >= 0) & (ih < H)
            for kw in tl.static_range(3):
                iw = w2 + kw - 1
                valid = vh & (iw >= 0) & (iw < W) & m2
                inp = tl.load(tmp_ptr + tc + ih * W_c + iw, mask=valid, other=0.0)
                wv = tl.load(w2_ptr + co2 * CMID * 9 + wc + kh * 3 + kw, mask=m2, other=0.0)
                acc2 += inp * wv
    acc2 += tl.load(b2_ptr + co2, mask=m2, other=0.0)
    acc2 = tl.maximum(acc2, 0.0)
    ob = n * Cout * HW
    tl.store(out_ptr + ob + offs2, acc2, mask=m2)


class ConvRelu2New(nn.Module):
    def __init__(self, _in, _out):
        super().__init__()
        self.cr1 = _CR(_in, _out)
        self.cr2 = _CR(_out, _out)

    def forward(self, x):
        x = x.contiguous()
        N, Cin, H, W = x.shape
        Cmid = self.cr1.conv.weight.shape[0]
        Cout = self.cr2.conv.weight.shape[0]
        HW = H * W
        tmp = torch.empty((N, Cmid, H, W), device=x.device, dtype=x.dtype)
        out = torch.empty((N, Cout, H, W), device=x.device, dtype=x.dtype)
        _two_conv_relu_kernel[(N,)](
            x, self.cr1.conv.weight, self.cr1.conv.bias,
            self.cr2.conv.weight, self.cr2.conv.bias, tmp, out,
            N, Cin, Cmid, Cout, H, W,
            HW=HW, W_c=W, CIN=Cin, CMID=Cmid,
            BLOCK1=triton.next_power_of_2(Cmid * HW),
            BLOCK2=triton.next_power_of_2(Cout * HW),
            num_warps=2,
        )
        return out


class _CR(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)
