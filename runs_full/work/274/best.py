import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import ReLU, Conv2d
from torch.nn.modules.utils import _pair
import triton
import triton.language as tl


@triton.jit
def _conv2d_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                   N, IC, IH, IW, OC, OH, OW, OHW,
                   stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                   KH: tl.constexpr, KW: tl.constexpr,
                   ICg: tl.constexpr, OCg: tl.constexpr,
                   HAS_BIAS: tl.constexpr, RELU: tl.constexpr,
                   BLOCK: tl.constexpr):
    pid_nc = tl.program_id(0)
    pid_s = tl.program_id(1)
    n = pid_nc // OC
    oc = pid_nc % OC
    group = oc // OCg
    ic_base = group * ICg
    offs = pid_s * BLOCK + tl.arange(0, BLOCK)
    mask = offs < OHW
    oh = offs // OW
    ow = offs % OW
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    if HAS_BIAS:
        acc += tl.load(b_ptr + oc)
    for ic in range(ICg):
        cin = ic_base + ic
        for kh in range(KH):
            ih = oh * stride_h - pad_h + kh * dil_h
            for kw in range(KW):
                iw = ow * stride_w - pad_w + kw * dil_w
                valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
                xidx = ((n * IC + cin) * IH + ih) * IW + iw
                xv = tl.load(x_ptr + xidx, mask=valid, other=0.0)
                widx = ((oc * ICg + ic) * KH + kh) * KW + kw
                wv = tl.load(w_ptr + widx)
                acc += xv * wv
    if RELU:
        acc = tl.maximum(acc, 0.0)
    oidx = (n * OC + oc) * OHW + offs
    tl.store(out_ptr + oidx, acc, mask=mask)


def conv2d_triton(x, weight, bias, stride, padding, dilation, groups, relu):
    x = x.contiguous(); weight = weight.contiguous()
    N, IC, IH, IW = x.shape
    OC, ICg, KH, KW = weight.shape
    sh, sw = stride; ph, pw = padding; dh, dw = dilation
    OH = (IH + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    OW = (IW + 2 * pw - dw * (KW - 1) - 1) // sw + 1
    OHW = OH * OW
    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)
    OCg = OC // groups
    has_bias = bias is not None
    if not has_bias:
        bias = x
    BLOCK = 256
    grid = (N * OC, triton.cdiv(OHW, BLOCK))
    _conv2d_kernel[grid](x, weight, bias, out, N, IC, IH, IW, OC, OH, OW, OHW,
                         sh, sw, ph, pw, dh, dw, KH=KH, KW=KW, ICg=ICg, OCg=OCg,
                         HAS_BIAS=has_bias, RELU=relu, BLOCK=BLOCK, num_warps=4)
    return out


@triton.jit
def _gap_kernel(x_ptr, out_ptr, N, C, RC, OHW,
                RADIX: tl.constexpr, BLOCK_S: tl.constexpr):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    offs = tl.arange(0, BLOCK_S)
    mask = offs < OHW
    acc = tl.zeros((), dtype=tl.float32)
    for r in range(RADIX):
        ch = r * C + c
        base = (n * RC + ch) * OHW
        v = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc += tl.sum(v)
    tl.store(out_ptr + (n * C + c), acc / OHW)


def gap_triton(x, channels, radix):
    x = x.contiguous()
    N, RC, H, W = x.shape
    OHW = H * W
    out = torch.empty((N, channels, 1, 1), device=x.device, dtype=x.dtype)
    BLOCK_S = max(1, triton.next_power_of_2(OHW))
    grid = (N * channels,)
    _gap_kernel[grid](x, out, N, channels, RC, OHW, RADIX=radix, BLOCK_S=BLOCK_S, num_warps=4)
    return out


@triton.jit
def _fc2_row(w2_ptr, b2_ptr, h, r, C, INTER, cidx, j, cmask, jmask):
    o = r * C + cidx
    w2 = tl.load(w2_ptr + o[:, None] * INTER + j[None, :],
                 mask=cmask[:, None] & jmask[None, :], other=0.0)
    return tl.sum(w2 * h[None, :], axis=1) + tl.load(b2_ptr + o, mask=cmask, other=0.0)


@triton.jit
def _attn_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
                 C, INTER, RC, OHW,
                 C_B: tl.constexpr, INTER_B: tl.constexpr, RADIX: tl.constexpr,
                 S_B: tl.constexpr):
    n = tl.program_id(0)
    cidx = tl.arange(0, C_B)
    cmask = cidx < C
    # gap[c] = mean_spatial( sum_r x[n, r*C+c, :] )
    sidx = tl.arange(0, S_B)
    smask = sidx < OHW
    g = tl.zeros((C_B,), dtype=tl.float32)
    for r in range(RADIX):
        ch = r * C + cidx
        addr = (n * RC + ch)[:, None] * OHW + sidx[None, :]
        xv = tl.load(x_ptr + addr, mask=cmask[:, None] & smask[None, :], other=0.0)
        g += tl.sum(xv, axis=1)
    g = g / OHW  # [C_B]
    # fc1: h[j] = relu( sum_c W1[j,c]*g[c] + b1[j] )
    j = tl.arange(0, INTER_B)
    jmask = j < INTER
    w1 = tl.load(w1_ptr + j[:, None] * C + cidx[None, :],
                 mask=jmask[:, None] & cmask[None, :], other=0.0)  # [INTER_B,C_B]
    h = tl.sum(w1 * g[None, :], axis=1) + tl.load(b1_ptr + j, mask=jmask, other=0.0)
    h = tl.maximum(h, 0.0)  # [INTER_B]
    # softmax over radix per channel k (recompute a_r each pass, tiny)
    m = tl.full((C_B,), -float('inf'), dtype=tl.float32)
    for r in range(RADIX):
        m = tl.maximum(m, _fc2_row(w2_ptr, b2_ptr, h, r, C, INTER, cidx, j, cmask, jmask))
    s = tl.zeros((C_B,), dtype=tl.float32)
    for r in range(RADIX):
        s += tl.exp(_fc2_row(w2_ptr, b2_ptr, h, r, C, INTER, cidx, j, cmask, jmask) - m)
    # fused combine: out[n,c,sp] = sum_r softmax_r[c] * x[n, r*C+c, sp]
    acc = tl.zeros((C_B, S_B), dtype=tl.float32)
    for r in range(RADIX):
        a_r = _fc2_row(w2_ptr, b2_ptr, h, r, C, INTER, cidx, j, cmask, jmask)
        sm = tl.exp(a_r - m) / s  # [C_B]
        ch = r * C + cidx
        addr = (n * RC + ch)[:, None] * OHW + sidx[None, :]
        xv = tl.load(x_ptr + addr, mask=cmask[:, None] & smask[None, :], other=0.0)
        acc += sm[:, None] * xv
    oaddr = (n * C + cidx)[:, None] * OHW + sidx[None, :]
    tl.store(out_ptr + oaddr, acc, mask=cmask[:, None] & smask[None, :])


def attn_triton(x, w1, b1, w2, b2, channels, inter, radix):
    x = x.contiguous()
    N, RC, H, W = x.shape
    OHW = H * W
    out = torch.empty((N, channels, H, W), device=x.device, dtype=x.dtype)
    C_B = max(1, triton.next_power_of_2(channels))
    INTER_B = max(1, triton.next_power_of_2(inter))
    S_B = max(1, triton.next_power_of_2(OHW))
    grid = (N,)
    _attn_kernel[grid](x, w1.contiguous(), b1.contiguous(),
                       w2.contiguous(), b2.contiguous(), out,
                       channels, inter, RC, OHW, C_B=C_B, INTER_B=INTER_B,
                       S_B=S_B,
                       RADIX=radix, num_warps=4)
    return out


@triton.jit
def _combine_kernel(x_ptr, a_ptr, out_ptr, N, channels, RC, OHW,
                    RADIX: tl.constexpr, BLOCK: tl.constexpr):
    pid_nc = tl.program_id(0)
    pid_s = tl.program_id(1)
    n = pid_nc // channels
    c = pid_nc % channels
    offs = pid_s * BLOCK + tl.arange(0, BLOCK)
    mask = offs < OHW
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for r in range(RADIX):
        a = tl.load(a_ptr + (n * RC + r * channels + c))
        ch = r * channels + c
        xv = tl.load(x_ptr + (n * RC + ch) * OHW + offs, mask=mask, other=0.0)
        acc += a * xv
    tl.store(out_ptr + (n * channels + c) * OHW + offs, acc, mask=mask)


def combine_triton(x, atten_sm, channels, radix):
    x = x.contiguous()
    N, RC, H, W = x.shape
    OHW = H * W
    out = torch.empty((N, channels, H, W), device=x.device, dtype=x.dtype)
    BLOCK = 256
    grid = (N * channels, triton.cdiv(OHW, BLOCK))
    _combine_kernel[grid](x, atten_sm, out, N, channels, RC, OHW,
                          RADIX=radix, BLOCK=BLOCK, num_warps=4)
    return out


@triton.jit
def _fused_kernel(x_ptr, cw_ptr, cb_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
                  N, IC, IH, IW, C, INTER, OH, OW, OHW,
                  stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                  RADIX: tl.constexpr, C_B: tl.constexpr, INTER_B: tl.constexpr,
                  S_B: tl.constexpr, ICg: tl.constexpr, OCg: tl.constexpr,
                  KH: tl.constexpr, KW: tl.constexpr):
    n = tl.program_id(0)
    rr = tl.arange(0, RADIX)          # [R]
    cidx = tl.arange(0, C_B)          # [C_B]
    sidx = tl.arange(0, S_B)          # [S_B]
    cmask = cidx < C
    smask = sidx < OHW
    oc = rr[:, None] * C + cidx[None, :]          # [R, C_B]
    ocmask = cmask[None, :]                        # [R, C_B] (rr always valid)
    RC = RADIX * C

    oh = sidx // OW
    ow = sidx % OW

    # grouped conv (groups = cardinality*radix, cardinality==1 -> group == r)
    acc = tl.load(cb_ptr + oc, mask=ocmask, other=0.0)[:, :, None] + \
        tl.zeros((RADIX, C_B, S_B), dtype=tl.float32)
    ic_base = (oc // OCg) * ICg                    # [R, C_B]
    for ic in range(ICg):
        cin = ic_base + ic                          # [R, C_B]
        for kh in range(KH):
            ih = oh * stride_h - pad_h + kh * dil_h
            for kw in range(KW):
                iw = ow * stride_w - pad_w + kw * dil_w
                valid = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW) & smask
                xaddr = (n * IC + cin)[:, :, None] * (IH * IW) + (ih * IW + iw)[None, None, :]
                vmask = ocmask[:, :, None] & valid[None, None, :]
                xv = tl.load(x_ptr + xaddr, mask=vmask, other=0.0)
                widx = (oc * ICg + ic) * (KH * KW) + kh * KW + kw
                w = tl.load(cw_ptr + widx, mask=ocmask, other=0.0)
                acc += xv * w[:, :, None]
    acc = tl.maximum(acc, 0.0)                       # x_conv [R, C_B, S_B]

    # gap[c] = mean_spatial( sum_r x_conv[r,c,:] )
    g = tl.sum(tl.sum(acc, axis=0), axis=1) / OHW    # [C_B]

    # fc1: h = relu(W1 @ g + b1)
    j = tl.arange(0, INTER_B)
    jmask = j < INTER
    w1 = tl.load(w1_ptr + j[:, None] * C + cidx[None, :],
                 mask=jmask[:, None] & cmask[None, :], other=0.0)
    h = tl.sum(w1 * g[None, :], axis=1) + tl.load(b1_ptr + j, mask=jmask, other=0.0)
    h = tl.maximum(h, 0.0)                            # [INTER_B]

    # fc2: a[r,c] = sum_j W2[r*C+c, j]*h[j] + b2
    w2 = tl.load(w2_ptr + oc[:, :, None] * INTER + j[None, None, :],
                 mask=ocmask[:, :, None] & jmask[None, None, :], other=0.0)
    a = tl.sum(w2 * h[None, None, :], axis=2) + tl.load(b2_ptr + oc, mask=ocmask, other=0.0)

    # softmax over radix axis (axis 0)
    m = tl.max(a, axis=0)                             # [C_B]
    e = tl.exp(a - m[None, :])
    s = tl.sum(e, axis=0)                             # [C_B]
    sm = e / s[None, :]                               # [R, C_B]

    # combine: out[c,sp] = sum_r sm[r,c] * x_conv[r,c,sp]
    out2d = tl.sum(sm[:, :, None] * acc, axis=0)      # [C_B, S_B]
    oaddr = (n * C + cidx)[:, None] * OHW + sidx[None, :]
    tl.store(out_ptr + oaddr, out2d, mask=cmask[:, None] & smask[None, :])


def fused_triton(x, cw, cb, w1, b1, w2, b2, channels, inter, radix, cardinality,
                 stride, padding, dilation):
    x = x.contiguous()
    N, IC, IH, IW = x.shape
    OC, ICg, KH, KW = cw.shape          # OC = channels*radix
    sh, sw = stride; ph, pw = padding; dh, dw = dilation
    OH = (IH + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    OW = (IW + 2 * pw - dw * (KW - 1) - 1) // sw + 1
    OHW = OH * OW
    groups_conv = cardinality * radix
    OCg = OC // groups_conv
    out = torch.empty((N, channels, OH, OW), device=x.device, dtype=x.dtype)
    C_B = max(1, triton.next_power_of_2(channels))
    INTER_B = max(1, triton.next_power_of_2(inter))
    S_B = max(1, triton.next_power_of_2(OHW))
    grid = (N,)
    _fused_kernel[grid](x, cw.contiguous(), cb.contiguous(),
                        w1.contiguous(), b1.contiguous(),
                        w2.contiguous(), b2.contiguous(), out,
                        N, IC, IH, IW, channels, inter, OH, OW, OHW,
                        sh, sw, ph, pw, dh, dw,
                        RADIX=radix, C_B=C_B, INTER_B=INTER_B, S_B=S_B,
                        ICg=ICg, OCg=OCg, KH=KH, KW=KW, num_warps=1)
    return out


def get_norm(norm, out_channels, **kwargs):
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
    return norm(out_channels, **kwargs)


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality


class SplAtConv2dNew(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1),
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=True, radix=2,
                 reduction_factor=4, rectify=False, rectify_avg=False,
                 norm_layer=None, dropblock_prob=0.0, **kwargs):
        super().__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.inter_channels = inter_channels
        self.dropblock_prob = dropblock_prob
        self.stride = _pair(stride)
        self.padding = padding
        self.dilation = _pair(dilation)
        self.conv = Conv2d(in_channels, channels * radix, kernel_size,
                           stride, padding, dilation, groups=groups * radix,
                           bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = get_norm(norm_layer, channels * radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = get_norm(norm_layer, inter_channels)
        self.fc2 = Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        channels = self.channels
        r = self.radix
        # fully-fused single-launch fast path
        if (r > 1 and self.cardinality == 1 and not self.use_bn
                and (r & (r - 1)) == 0 and self.conv.bias is not None):
            out = fused_triton(x, self.conv.weight, self.conv.bias,
                               self.fc1.weight, self.fc1.bias,
                               self.fc2.weight, self.fc2.bias,
                               channels, self.inter_channels, r,
                               self.cardinality, self.stride, self.padding,
                               self.dilation)
            return out.contiguous()
        x = conv2d_triton(x, self.conv.weight, self.conv.bias,
                          self.stride, self.padding, self.dilation,
                          self.cardinality * self.radix, relu=not self.use_bn)
        if self.use_bn:
            x = self.bn0(x); x = self.relu(x)
        if self.radix > 1 and self.cardinality == 1:
            out = attn_triton(x, self.fc1.weight, self.fc1.bias,
                              self.fc2.weight, self.fc2.bias,
                              channels, self.inter_channels, self.radix)
            return out.contiguous()
        # fallback path
        if self.radix > 1:
            gap = gap_triton(x, channels, self.radix)
        else:
            gap = F.adaptive_avg_pool2d(x, 1)
        gap = conv2d_triton(gap, self.fc1.weight, self.fc1.bias,
                            (1, 1), (0, 0), (1, 1), self.cardinality, relu=not self.use_bn)
        if self.use_bn:
            gap = self.bn1(gap); gap = self.relu(gap)
        atten = conv2d_triton(gap, self.fc2.weight, self.fc2.bias,
                              (1, 1), (0, 0), (1, 1), self.cardinality, relu=False)
        if self.radix > 1:
            atten = atten.view(atten.shape[0], self.cardinality, self.radix, -1).transpose(1, 2)
            atten = F.softmax(atten, dim=1).reshape(atten.shape[0], -1)
            atten_sm = atten.view(atten.shape[0], -1)
            out = combine_triton(x, atten_sm.contiguous(), channels, self.radix)
        else:
            atten = torch.sigmoid(atten)
            out = atten * x
        return out.contiguous()
