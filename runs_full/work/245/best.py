import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                N, IC, OC, H, W,
                KH: tl.constexpr, KW: tl.constexpr, PAD: tl.constexpr,
                BLOCK_W: tl.constexpr, BLOCK_OC: tl.constexpr, BLOCK_IC: tl.constexpr,
                RELU: tl.constexpr, HAS_BIAS: tl.constexpr):
    pid_nh = tl.program_id(0)
    pid_oc = tl.program_id(1)
    n = pid_nh // H
    oh = pid_nh % H
    offs_w = tl.arange(0, BLOCK_W)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    w_mask = offs_w < W
    oc_mask = offs_oc < OC
    acc = tl.zeros((BLOCK_W, BLOCK_OC), tl.float32)
    for kh in range(KH):
        ih = oh + kh - PAD
        ih_valid = (ih >= 0) & (ih < H)
        for kw in range(KW):
            iw = offs_w + kw - PAD
            iw_valid = (iw >= 0) & (iw < W)
            for ic0 in range(0, IC, BLOCK_IC):
                offs_ic = ic0 + tl.arange(0, BLOCK_IC)
                ic_mask = offs_ic < IC
                x_addr = ((n * IC + offs_ic[None, :]) * H + ih) * W + iw[:, None]
                xmask = iw_valid[:, None] & ih_valid & ic_mask[None, :] & w_mask[:, None]
                x_tile = tl.load(x_ptr + x_addr, mask=xmask, other=0.0)
                w_addr = ((offs_oc[None, :] * IC + offs_ic[:, None]) * KH + kh) * KW + kw
                wmask = ic_mask[:, None] & oc_mask[None, :]
                w_tile = tl.load(w_ptr + w_addr, mask=wmask, other=0.0)
                acc += tl.dot(x_tile, w_tile)
    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_oc, mask=oc_mask, other=0.0)
        acc += bias[None, :]
    if RELU:
        acc = tl.maximum(acc, 0.0)
    out_addr = ((n * OC + offs_oc[None, :]) * H + oh) * W + offs_w[:, None]
    omask = w_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_addr, acc, mask=omask)


@triton.jit
def maxpool_kernel(x_ptr, out_ptr, N, C, H, W, OH, OW, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * OH * OW
    mask = offs < total
    ow = offs % OW
    t = offs // OW
    oh = t % OH
    t2 = t // OH
    c = t2 % C
    n = t2 // C
    ih = oh * 2
    iw = ow * 2
    base = (n * C + c) * H
    a = tl.load(x_ptr + (base + ih) * W + iw, mask=mask, other=-float('inf'))
    b = tl.load(x_ptr + (base + ih) * W + iw + 1, mask=mask, other=-float('inf'))
    cc = tl.load(x_ptr + (base + ih + 1) * W + iw, mask=mask, other=-float('inf'))
    d = tl.load(x_ptr + (base + ih + 1) * W + iw + 1, mask=mask, other=-float('inf'))
    m = tl.maximum(tl.maximum(a, b), tl.maximum(cc, d))
    tl.store(out_ptr + offs, m, mask=mask)


@triton.jit
def bilinear_kernel(x_ptr, out_ptr, N, C, H, W, OH, OW, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * OH * OW
    mask = offs < total
    ow = offs % OW
    t = offs // OW
    oh = t % OH
    t2 = t // OH
    c = t2 % C
    n = t2 // C
    fh = 0.5 * (oh.to(tl.float32) + 0.5) - 0.5
    fh = tl.maximum(fh, 0.0)
    fw = 0.5 * (ow.to(tl.float32) + 0.5) - 0.5
    fw = tl.maximum(fw, 0.0)
    h0 = fh.to(tl.int32)
    w0 = fw.to(tl.int32)
    h1 = tl.where(h0 < H - 1, h0 + 1, h0)
    w1 = tl.where(w0 < W - 1, w0 + 1, w0)
    lh1 = fh - h0.to(tl.float32)
    lh0 = 1.0 - lh1
    lw1 = fw - w0.to(tl.float32)
    lw0 = 1.0 - lw1
    base = (n * C + c) * H
    a = tl.load(x_ptr + (base + h0) * W + w0, mask=mask, other=0.0)
    b = tl.load(x_ptr + (base + h0) * W + w1, mask=mask, other=0.0)
    cc = tl.load(x_ptr + (base + h1) * W + w0, mask=mask, other=0.0)
    d = tl.load(x_ptr + (base + h1) * W + w1, mask=mask, other=0.0)
    out = lh0 * lw0 * a + lh0 * lw1 * b + lh1 * lw0 * cc + lh1 * lw1 * d
    tl.store(out_ptr + offs, out, mask=mask)


def triton_conv(x, weight, bias, relu, pad, kh, kw):
    x = x.contiguous()
    N, IC, H, W = x.shape
    OC = weight.shape[0]
    out = torch.empty((N, OC, H, W), dtype=x.dtype, device=x.device)
    BLOCK_W = triton.next_power_of_2(W)
    if BLOCK_W < 16:
        BLOCK_W = 16
    BLOCK_OC = 64
    BLOCK_IC = 64
    grid = (N * H, triton.cdiv(OC, BLOCK_OC))
    b = bias if bias is not None else x
    conv_kernel[grid](x, weight, b, out, N, IC, OC, H, W,
                      kh, kw, pad, BLOCK_W, BLOCK_OC, BLOCK_IC,
                      relu, bias is not None, num_warps=4, num_stages=2)
    return out


def triton_maxpool(x):
    x = x.contiguous()
    N, C, H, W = x.shape
    OH, OW = H // 2, W // 2
    out = torch.empty((N, C, OH, OW), dtype=x.dtype, device=x.device)
    total = N * C * OH * OW
    BLOCK = 256
    grid = (triton.cdiv(total, BLOCK),)
    maxpool_kernel[grid](x, out, N, C, H, W, OH, OW, BLOCK, num_warps=4)
    return out


def triton_bilinear(x):
    x = x.contiguous()
    N, C, H, W = x.shape
    OH, OW = H * 2, W * 2
    out = torch.empty((N, C, OH, OW), dtype=x.dtype, device=x.device)
    total = N * C * OH * OW
    BLOCK = 256
    grid = (triton.cdiv(total, BLOCK),)
    bilinear_kernel[grid](x, out, N, C, H, W, OH, OW, BLOCK, num_warps=4)
    return out


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x))


class ConvRelu2(nn.Module):
    def __init__(self, _in, _out):
        super(ConvRelu2, self).__init__()
        self.cr1 = ConvRelu(_in, _out)
        self.cr2 = ConvRelu(_out, _out)

    def forward(self, x):
        return self.cr2(self.cr1(x))


class Coder(nn.Module):
    def __init__(self, in_size, out_size):
        super(Coder, self).__init__()
        self.conv = ConvRelu2(in_size, out_size)
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, x):
        y1 = self.conv(x)
        return self.down(y1), y1


class Decoder(nn.Module):
    def __init__(self, in_size, out_size):
        super(Decoder, self).__init__()
        self.conv = ConvRelu2(in_size, out_size)

    def forward(self, x1, x2):
        pass


def _cr(x, cr):
    return triton_conv(x, cr.conv.weight, cr.conv.bias, True, 1, 3, 3)


class AttentionNetNew(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(AttentionNetNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        filters = [64, 128, 256]
        self.down1 = Coder(in_channels, filters[0])
        self.down2 = Coder(filters[0], filters[1])
        self.center = ConvRelu2(filters[1], filters[2])
        self.up2 = Decoder(filters[2] + filters[1], filters[1])
        self.up1 = Decoder(filters[1] + filters[0], filters[0])
        self.final = nn.Conv2d(filters[0], out_channels, 1)

    def forward(self, x):
        y1 = _cr(_cr(x, self.down1.conv.cr1), self.down1.conv.cr2)
        befdown1 = y1
        x = triton_maxpool(y1)
        y2 = _cr(_cr(x, self.down2.conv.cr1), self.down2.conv.cr2)
        befdown2 = y2
        x = triton_maxpool(y2)
        c = _cr(_cr(x, self.center.cr1), self.center.cr2)
        up = triton_bilinear(c)
        cat = torch.cat([befdown2, up], 1)
        x = _cr(_cr(cat, self.up2.conv.cr1), self.up2.conv.cr2)
        up = triton_bilinear(x)
        cat = torch.cat([befdown1, up], 1)
        x = _cr(_cr(cat, self.up1.conv.cr1), self.up1.conv.cr2)
        x = triton_conv(x, self.final.weight, self.final.bias, False, 0, 1, 1)
        return x
