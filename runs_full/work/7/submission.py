import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                 N, IH, IW, OC, OH, OW, SH, SW, PH, PW,
                 total,
                 IC: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
                 HAS_BIAS: tl.constexpr, RELU: tl.constexpr,
                 BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    ow = offs % OW
    t = offs // OW
    oh = t % OH
    t = t // OH
    oc = t % OC
    n = t // OC

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for ic in tl.static_range(IC):
        for kh in tl.static_range(KH):
            ih = oh * SH - PH + kh
            for kw in tl.static_range(KW):
                iw = ow * SW - PW + kw
                valid = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
                x_off = n * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw
                xv = tl.load(x_ptr + x_off, mask=mask & valid, other=0.0)
                w_off = oc * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                wv = tl.load(w_ptr + w_off, mask=mask, other=0.0)
                acc += xv * wv

    if HAS_BIAS:
        acc += tl.load(b_ptr + oc, mask=mask, other=0.0)
    if RELU:
        acc = tl.maximum(acc, 0.0)

    tl.store(out_ptr + offs, acc, mask=mask)


@triton.jit
def _conv1x1_add_relu_kernel(x_ptr, w_ptr, s_ptr, out_ptr,
                             N, OC, OHr, OWr, OH3, OW3, SH_, SW_,
                             total,
                             IC: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    ow = offs % OWr
    t = offs // OWr
    oh = t % OHr
    t = t // OHr
    oc = t % OC
    n = t // OC

    # broadcast indices
    c3h = tl.where(OH3 > 1, oh, 0)
    c3w = tl.where(OW3 > 1, ow, 0)
    sh = tl.where(SH_ > 1, oh, 0)
    sw = tl.where(SW_ > 1, ow, 0)

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for ic in tl.static_range(IC):
        x_off = n * (IC * OH3 * OW3) + ic * (OH3 * OW3) + c3h * OW3 + c3w
        xv = tl.load(x_ptr + x_off, mask=mask, other=0.0)
        wv = tl.load(w_ptr + oc * IC + ic, mask=mask, other=0.0)
        acc += xv * wv

    s_off = n * (OC * SH_ * SW_) + oc * (SH_ * SW_) + sh * SW_ + sw
    sv = tl.load(s_ptr + s_off, mask=mask, other=0.0)
    acc = tl.maximum(acc + sv, 0.0)
    tl.store(out_ptr + offs, acc, mask=mask)


@triton.jit
def _fused_c1c2_kernel(x_ptr, w1_ptr, w2_ptr, b2_ptr, out_ptr,
                       N, IH, IW, OH, OW, SH, SW,
                       total,
                       INDIM: tl.constexpr, BDIM: tl.constexpr,
                       KH: tl.constexpr, KW: tl.constexpr,
                       BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    ow = offs % OW
    t = offs // OW
    oh = t % OH
    t = t // OH
    oc2 = t % BDIM
    n = t // BDIM

    acc = tl.load(b2_ptr + oc2, mask=mask, other=0.0)
    for kh in tl.static_range(KH):
        ih = oh * SH - 1 + kh
        for kw in tl.static_range(KW):
            iw = ow * SW - 1 + kw
            valid = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            for ic2 in tl.static_range(BDIM):
                c1 = tl.zeros((BLOCK,), dtype=tl.float32)
                for c in tl.static_range(INDIM):
                    x_off = n * (INDIM * IH * IW) + c * (IH * IW) + ih * IW + iw
                    xv = tl.load(x_ptr + x_off, mask=mask & valid, other=0.0)
                    w1v = tl.load(w1_ptr + ic2 * INDIM + c)
                    c1 += xv * w1v
                c1 = tl.maximum(c1, 0.0)
                w2_off = oc2 * (BDIM * KH * KW) + ic2 * (KH * KW) + kh * KW + kw
                w2v = tl.load(w2_ptr + w2_off, mask=mask, other=0.0)
                acc += c1 * w2v

    acc = tl.maximum(acc, 0.0)
    tl.store(out_ptr + offs, acc, mask=mask)


@triton.jit
def _fused_all_kernel(x_ptr, w1_ptr, w2_ptr, b2_ptr, w3_ptr, s_ptr, out_ptr,
                      N, IH, IW, OH3, OW3, OHr, OWr, OUTDIM, SH_, SW_, S,
                      total,
                      INDIM: tl.constexpr, BDIM: tl.constexpr,
                      KH: tl.constexpr, KW: tl.constexpr,
                      BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    ow = offs % OWr
    t = offs // OWr
    oh = t % OHr
    t = t // OHr
    oc = t % OUTDIM
    n = t // OUTDIM

    c3h = tl.where(OH3 > 1, oh, 0)
    c3w = tl.where(OW3 > 1, ow, 0)
    sh = tl.where(SH_ > 1, oh, 0)
    sw = tl.where(SW_ > 1, ow, 0)

    c3acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for bc in tl.static_range(BDIM):
        c2 = tl.load(b2_ptr + bc)
        c2 = tl.broadcast_to(c2, (BLOCK,)).to(tl.float32)
        for kh in tl.static_range(KH):
            ih = c3h * S - 1 + kh
            for kw in tl.static_range(KW):
                iw = c3w * S - 1 + kw
                valid = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
                for m in tl.static_range(BDIM):
                    c1 = tl.zeros((BLOCK,), dtype=tl.float32)
                    for c in tl.static_range(INDIM):
                        x_off = n * (INDIM * IH * IW) + c * (IH * IW) + ih * IW + iw
                        xv = tl.load(x_ptr + x_off, mask=mask & valid, other=0.0)
                        w1v = tl.load(w1_ptr + m * INDIM + c)
                        c1 += xv * w1v
                    c1 = tl.maximum(c1, 0.0)
                    w2v = tl.load(w2_ptr + bc * (BDIM * KH * KW) + m * (KH * KW) + kh * KW + kw)
                    c2 += c1 * w2v
        c2 = tl.maximum(c2, 0.0)
        w3v = tl.load(w3_ptr + oc * BDIM + bc, mask=mask, other=0.0)
        c3acc += c2 * w3v

    s_off = n * (OUTDIM * SH_ * SW_) + oc * (SH_ * SW_) + sh * SW_ + sw
    sv = tl.load(s_ptr + s_off, mask=mask, other=0.0)
    out = tl.maximum(c3acc + sv, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


def _fused_all(x, w1, w2, b2, w3, short, stride):
    N, INDIM, IH, IW = x.shape
    BDIM = w2.shape[0]
    KH, KW = w2.shape[2], w2.shape[3]
    OUTDIM = w3.shape[0]
    S = stride
    OH3 = (IH + 2 - KH) // S + 1
    OW3 = (IW + 2 - KW) // S + 1
    _, _, SH_, SW_ = short.shape
    OHr = max(OH3, SH_)
    OWr = max(OW3, SW_)
    out = torch.empty((N, OUTDIM, OHr, OWr), device=x.device, dtype=torch.float32)
    total = N * OUTDIM * OHr * OWr
    BLOCK = 256
    grid = (triton.cdiv(total, BLOCK),)
    _fused_all_kernel[grid](
        x, w1, w2, b2, w3, short, out,
        N, IH, IW, OH3, OW3, OHr, OWr, OUTDIM, SH_, SW_, S, total,
        INDIM=INDIM, BDIM=BDIM, KH=KH, KW=KW,
        BLOCK=BLOCK, num_warps=4,
    )
    return out


def _fused_c1c2(x, w1, w2, b2, stride):
    N, INDIM, IH, IW = x.shape
    BDIM = w2.shape[0]
    KH, KW = w2.shape[2], w2.shape[3]
    SH = SW = stride
    OH = (IH + 2 - KH) // SH + 1
    OW = (IW + 2 - KW) // SW + 1
    out = torch.empty((N, BDIM, OH, OW), device=x.device, dtype=torch.float32)
    total = N * BDIM * OH * OW
    BLOCK = 256
    grid = (triton.cdiv(total, BLOCK),)
    _fused_c1c2_kernel[grid](
        x, w1, w2, b2, out,
        N, IH, IW, OH, OW, SH, SW, total,
        INDIM=INDIM, BDIM=BDIM, KH=KH, KW=KW,
        BLOCK=BLOCK, num_warps=4,
    )
    return out


def _conv1x1_add_relu(x, weight, short):
    N, IC, OH3, OW3 = x.shape
    OC = weight.shape[0]
    _, _, SH_, SW_ = short.shape
    OHr = max(OH3, SH_)
    OWr = max(OW3, SW_)
    out = torch.empty((N, OC, OHr, OWr), device=x.device, dtype=torch.float32)
    total = N * OC * OHr * OWr
    BLOCK = 256
    grid = (triton.cdiv(total, BLOCK),)
    _conv1x1_add_relu_kernel[grid](
        x, weight, short, out,
        N, OC, OHr, OWr, OH3, OW3, SH_, SW_, total,
        IC=IC, BLOCK=BLOCK, num_warps=4,
    )
    return out


def _conv(x, weight, bias, stride, padding, relu):
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = weight.shape
    SH = SW = stride
    PH = PW = padding
    OH = (IH + 2 * PH - KH) // SH + 1
    OW = (IW + 2 * PW - KW) // SW + 1
    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)
    total = N * OC * OH * OW
    BLOCK = 256
    grid = (triton.cdiv(total, BLOCK),)
    _conv_kernel[grid](
        x, weight, bias if bias is not None else x, out,
        N, IH, IW, OC, OH, OW, SH, SW, PH, PW, total,
        IC=IC, KH=KH, KW=KW,
        HAS_BIAS=bias is not None, RELU=relu,
        BLOCK=BLOCK, num_warps=4,
    )
    return out


def init_layer(L):
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class BottleneckBlockNew(nn.Module):

    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlockNew, self).__init__()
        bottleneckdim = int(outdim / 4)
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1, bias=False)
        self.BN1 = nn.Identity()
        self.C2 = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=3,
            stride=2 if half_res else 1, padding=1)
        self.BN2 = nn.Identity()
        self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
        self.BN3 = nn.Identity()
        self.relu = nn.ReLU()
        self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2,
            self.C3, self.BN3]
        self.half_res = half_res
        if indim != outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, stride=2 if
                half_res else 1, bias=False)
            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'
        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        x = x.contiguous()
        if self.shortcut_type == 'identity':
            short_out = x
        else:
            s = self.shortcut.stride[0]
            short_out = _conv(x, self.shortcut.weight, None, s, 0, False)

        s2 = self.C2.stride[0]
        out = _fused_all(x, self.C1.weight, self.C2.weight, self.C2.bias,
                         self.C3.weight, short_out, s2)
        return out
