import math
import torch
import torch.nn as nn
from numpy import prod
import triton
import triton.language as tl


def getLayerNormalizationFactor(x):
    size = x.weight.size()
    fan_in = prod(size[1:])
    return math.sqrt(2.0 / fan_in)


class ConstrainedLayer(nn.Module):
    def __init__(self, module, equalized=True, lrMul=1.0, initBiasToZero=True):
        super(ConstrainedLayer, self).__init__()
        self.module = module
        self.equalized = equalized
        if initBiasToZero:
            self.module.bias.data.fill_(0)
        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.module.weight.data /= lrMul
            self.weight = getLayerNormalizationFactor(self.module) * lrMul

    def forward(self, x):
        x = self.module(x)
        if self.equalized:
            x *= self.weight
        return x


class EqualizedLinear(ConstrainedLayer):
    def __init__(self, nChannelsPrevious, nChannels, bias=True, **kwargs):
        ConstrainedLayer.__init__(self, nn.Linear(nChannelsPrevious,
            nChannels, bias=bias), **kwargs)


@triton.jit
def _adain_fused(x_ptr, y_ptr, w_ptr, b_ptr, out_ptr,
                 C, S, IN, twoC, scale, eps,
                 BLOCK_IN: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    bidx = pid // C
    c = pid % C
    base = pid * S

    # fused linear for ya (row c) and yb (row C+c)
    k = tl.arange(0, BLOCK_IN)
    mk = k < IN
    yv = tl.load(y_ptr + bidx * IN + k, mask=mk, other=0.0)
    wa = tl.load(w_ptr + c * IN + k, mask=mk, other=0.0)
    wb = tl.load(w_ptr + (C + c) * IN + k, mask=mk, other=0.0)
    ya = (tl.sum(yv * wa, axis=0) + tl.load(b_ptr + c)) * scale
    yb = (tl.sum(yv * wb, axis=0) + tl.load(b_ptr + C + c)) * scale

    s = 0.0
    ss = 0.0
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < S
        v = tl.load(x_ptr + base + idx, mask=m, other=0.0)
        s += tl.sum(v, axis=0)
        ss += tl.sum(v * v, axis=0)
    mu = s / S
    var = tl.maximum(ss / S - mu * mu, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < S
        v = tl.load(x_ptr + base + idx, mask=m, other=0.0)
        tl.store(out_ptr + base + idx, ya * (v - mu) * rstd + yb, mask=m)


class AdaINNew(nn.Module):
    def __init__(self, dimIn, dimOut, epsilon=1e-08):
        super(AdaINNew, self).__init__()
        self.epsilon = epsilon
        self.styleModulator = EqualizedLinear(dimIn, 2 * dimOut, equalized=
            True, initBiasToZero=True)
        self.dimOut = dimOut

    def forward(self, x, y):
        batchSize, nChannel, width, height = x.size()
        x = x.contiguous()
        y = y.contiguous()
        twoC = 2 * self.dimOut
        IN = y.size(1)
        W = self.styleModulator.module.weight
        bias = self.styleModulator.module.bias
        scale = float(self.styleModulator.weight)
        out = torch.empty_like(x)
        S = width * height
        BC = batchSize * nChannel
        BLOCK = min(1024, triton.next_power_of_2(S))
        BLOCK_IN = triton.next_power_of_2(IN)
        _adain_fused[(BC,)](x, y, W, bias, out, nChannel, S, IN, twoC,
            scale, self.epsilon, BLOCK_IN=BLOCK_IN, BLOCK=BLOCK, num_warps=4)
        return out
