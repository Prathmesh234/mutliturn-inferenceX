import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _conv_mfm_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                     N, Hin, Win, Cout, Hout, Wout,
                     Cin: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
                     stride: tl.constexpr, pad: tl.constexpr,
                     BLOCK: tl.constexpr):
    pid = tl.program_id(0)        # over N * Cout
    pid_s = tl.program_id(1)      # spatial block
    n = pid // Cout
    cout = pid % Cout

    nspatial = Hout * Wout
    offs = pid_s * BLOCK + tl.arange(0, BLOCK)
    mask_s = offs < nspatial
    oh = offs // Wout
    ow = offs % Wout

    b0 = tl.load(b_ptr + cout)
    b1 = tl.load(b_ptr + cout + Cout)
    acc0 = tl.zeros((BLOCK,), tl.float32) + b0
    acc1 = tl.zeros((BLOCK,), tl.float32) + b1

    x_base = n * Cin * Hin * Win
    w0_base = cout * Cin * KH * KW
    w1_base = (cout + Cout) * Cin * KH * KW

    for cin in tl.static_range(Cin):
        for kh in tl.static_range(KH):
            ih = oh * stride - pad + kh
            ih_ok = (ih >= 0) & (ih < Hin)
            for kw in tl.static_range(KW):
                iw = ow * stride - pad + kw
                ok = mask_s & ih_ok & (iw >= 0) & (iw < Win)
                x_off = x_base + cin * Hin * Win + ih * Win + iw
                xval = tl.load(x_ptr + x_off, mask=ok, other=0.0)
                woff = cin * KH * KW + kh * KW + kw
                w0 = tl.load(w_ptr + w0_base + woff)
                w1 = tl.load(w_ptr + w1_base + woff)
                acc0 += xval * w0
                acc1 += xval * w1

    out = tl.maximum(acc0, acc1)
    out_off = (n * Cout + cout) * nspatial + offs
    tl.store(out_ptr + out_off, out, mask=mask_s)


def _conv_mfm(x, weight, bias, out_channels, stride, pad):
    N, Cin, Hin, Win = x.shape
    twoCout, _, KH, KW = weight.shape
    Cout = out_channels
    Hout = (Hin + 2 * pad - KH) // stride + 1
    Wout = (Win + 2 * pad - KW) // stride + 1
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    out = torch.empty((N, Cout, Hout, Wout), device=x.device, dtype=x.dtype)
    BLOCK = triton.next_power_of_2(Hout * Wout)
    grid = (N * Cout, triton.cdiv(Hout * Wout, BLOCK))
    _conv_mfm_kernel[grid](x, weight, bias, out,
                           N, Hin, Win, Cout, Hout, Wout,
                           Cin=Cin, KH=KH, KW=KW, stride=stride, pad=pad,
                           BLOCK=BLOCK, num_warps=2)
    return out


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2 * out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)
        self._stride = stride
        self._padding = padding

    def forward(self, x):
        return _conv_mfm(x, self.filter.weight, self.filter.bias,
                         self.out_channels, self._stride, self._padding)


class groupNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(groupNew, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x
