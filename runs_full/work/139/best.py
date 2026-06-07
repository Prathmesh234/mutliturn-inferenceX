import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_kernel(x_ptr, w_ptr, b_ptr, out_ptr, N, H, W,
                 C: tl.constexpr, KS: tl.constexpr, PAD: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    npix = N * H * W
    mask = offs < npix
    w = offs % W
    hh = (offs // W) % H
    n = offs // (H * W)
    acc = tl.zeros((BLOCK,), tl.float32)
    for kh in range(0, KS):
        for kw in range(0, KS):
            ih = hh + kh - PAD
            iw = w + kw - PAD
            vmask = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            nbase = n * C * H * W + ih * W + iw
            maxv = tl.full((BLOCK,), -float('inf'), tl.float32)
            sumv = tl.zeros((BLOCK,), tl.float32)
            for c in range(0, C):
                v = tl.load(x_ptr + nbase + c * H * W, mask=vmask, other=-float('inf'))
                maxv = tl.maximum(maxv, v)
                sumv += tl.where(vmask, v, 0.0)
            meanv = sumv / C
            w0 = tl.load(w_ptr + kh * KS + kw)
            w1 = tl.load(w_ptr + KS * KS + kh * KS + kw)
            acc += tl.where(vmask, w0 * maxv + w1 * meanv, 0.0)
    acc += tl.load(b_ptr)
    scale = tl.sigmoid(acc)
    xbase = n * C * H * W + hh * W + w
    for c in range(0, C):
        xv = tl.load(x_ptr + xbase + c * H * W, mask=mask, other=0.0)
        tl.store(out_ptr + xbase + c * H * W, xv * scale, mask=mask)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, relu=False, bn=False, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01,
                                 affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None


class ChannelPool(nn.Module):
    pass


class SpatialGateNew(nn.Module):
    def __init__(self):
        super(SpatialGateNew, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1,
                                 padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x = x.contiguous()
        N, C, H, W = x.shape
        KS = self.spatial.conv.kernel_size[0]
        PAD = self.spatial.conv.padding[0]
        wt = self.spatial.conv.weight.contiguous()
        bs = self.spatial.conv.bias.contiguous()
        out = torch.empty_like(x)
        npix = N * H * W
        BLOCK = 64
        grid = (triton.cdiv(npix, BLOCK),)
        fused_kernel[grid](x, wt, bs, out, N, H, W,
                           C=C, KS=KS, PAD=PAD, BLOCK=BLOCK, num_warps=2)
        return out
