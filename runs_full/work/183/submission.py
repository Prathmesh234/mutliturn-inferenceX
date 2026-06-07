import math
import torch
from torch import nn
import triton
import triton.language as tl


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        kernel = make_kernel(kernel) * factor ** 2
        self.register_buffer('kernel', kernel)
        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2
        self.pad = pad0, pad1

    def forward(self, input):
        raise NotImplementedError


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * upsample_factor ** 2
        self.register_buffer('kernel', kernel)
        self.pad = pad


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1,
                 activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.activation = activation
        self.scale = 1 / math.sqrt(in_dim) * lr_mul
        self.lr_mul = lr_mul


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim,
                 demodulate=True, upsample=False, downsample=False,
                 blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.eps = 1e-08
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        if upsample:
            factor = 2
            p = len(blur_kernel) - factor - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)
        if downsample:
            factor = 2
            p = len(blur_kernel) - factor + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))
        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel,
                                               kernel_size, kernel_size))
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate


@triton.jit
def _fused_kernel(input_ptr, weight_ptr, style_ptr, mw_ptr, mbias_ptr,
                  rgbbias_ptr, out_ptr, modconv_scale, scale_lin, P,
                  IN_CH: tl.constexpr, OUT_CH: tl.constexpr, S_DIM: tl.constexpr,
                  BLOCK_S: tl.constexpr, BLOCK_P: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // OUT_CH
    o = pid % OUT_CH
    offs = tl.arange(0, BLOCK_P)
    mask = offs < P
    offs_s = tl.arange(0, BLOCK_S)
    mask_s = offs_s < S_DIM
    style = tl.load(style_ptr + b * S_DIM + offs_s, mask=mask_s, other=0.0)
    acc = tl.zeros((BLOCK_P,), dtype=tl.float32)
    in_base = b * IN_CH * P
    w_base = o * IN_CH
    for c in range(IN_CH):
        mw = tl.load(mw_ptr + c * S_DIM + offs_s, mask=mask_s, other=0.0)
        mod_c = tl.sum(style * mw) * scale_lin + tl.load(mbias_ptr + c)
        w = modconv_scale * tl.load(weight_ptr + w_base + c) * mod_c
        x = tl.load(input_ptr + in_base + c * P + offs, mask=mask, other=0.0)
        acc += w * x
    acc += tl.load(rgbbias_ptr + o)
    tl.store(out_ptr + b * OUT_CH * P + o * P + offs, acc, mask=mask)


class ToRGBNew(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        if upsample:
            self.upsample = Upsample(blur_kernel)
        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        batch, in_channel, height, width = input.shape
        out_channel = self.conv.out_channel
        P = height * width
        input = input.contiguous()
        style = style.contiguous()

        S_DIM = self.conv.modulation.weight.shape[1]
        BLOCK_S = triton.next_power_of_2(S_DIM)
        BLOCK_P = triton.next_power_of_2(P)
        out = torch.empty((batch, out_channel, height, width), device=input.device, dtype=torch.float32)
        _fused_kernel[(batch * out_channel,)](
            input, self.conv.weight, style, self.conv.modulation.weight,
            self.conv.modulation.bias, self.bias, out,
            self.conv.scale, self.conv.modulation.scale, P,
            IN_CH=in_channel, OUT_CH=out_channel, S_DIM=S_DIM,
            BLOCK_S=BLOCK_S, BLOCK_P=BLOCK_P, num_warps=1)

        if skip is not None:
            out = out + self.upsample(skip)
        return out
