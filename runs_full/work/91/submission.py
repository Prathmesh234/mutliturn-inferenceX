import torch
import torch.nn as nn
import triton
import triton.language as tl


def _np2(x):
    n = 1
    while n < x:
        n *= 2
    return max(n, 1)


@triton.jit
def upsample_kernel(x_ptr, o_ptr, H, W, Hh, Ww, S, BLOCK_W: tl.constexpr):
    nc = tl.program_id(0)
    oh = tl.program_id(1)
    ow = tl.arange(0, BLOCK_W)
    mask = ow < Ww
    v = tl.load(x_ptr + nc * H * W + (oh // S) * W + (ow // S), mask=mask, other=0.0)
    tl.store(o_ptr + nc * Hh * Ww + oh * Ww + ow, v, mask=mask)


@triton.jit
def fused_block_kernel(pa, pb, pc, w1, b1, w2, b2, o_ptr, H, W, SLOPE,
                       CA: tl.constexpr, CB: tl.constexpr, CC: tl.constexpr,
                       CIN: tl.constexpr, CMID: tl.constexpr, COUT: tl.constexpr,
                       BLOCK_W: tl.constexpr):
    n = tl.program_id(0)
    h = tl.program_id(1)
    co = tl.program_id(2)
    ow = tl.arange(0, BLOCK_W)
    mask_w = ow < W
    HW = H * W
    acc = tl.zeros([BLOCK_W], tl.float32)
    for kh in tl.static_range(3):
        ih = h + kh - 1
        ih_ok = (ih >= 0) & (ih < H)
        for kw in tl.static_range(3):
            iw = ow + kw - 1
            m = mask_w & ih_ok & (iw >= 0) & (iw < W)
            roff = ih * W + iw
            for cm in tl.static_range(CMID):
                a = tl.zeros([BLOCK_W], tl.float32)
                for ci in tl.static_range(CIN):
                    if ci < CA:
                        x = tl.load(pa + n * CA * HW + ci * HW + roff, mask=m, other=0.0)
                    elif ci < CA + CB:
                        x = tl.load(pb + n * CB * HW + (ci - CA) * HW + roff, mask=m, other=0.0)
                    else:
                        x = tl.load(pc + n * CC * HW + (ci - CA - CB) * HW + roff, mask=m, other=0.0)
                    a += x * tl.load(w1 + cm * CIN + ci)
                a += tl.load(b1 + cm)
                a = tl.where(m, a, 0.0)
                acc += a * tl.load(w2 + ((co * CMID + cm) * 3 + kh) * 3 + kw)
    acc += tl.load(b2 + co)
    acc = tl.where(acc > 0, acc, acc * SLOPE)
    tl.store(o_ptr + n * COUT * HW + co * HW + h * W + ow, acc, mask=mask_w)


class DenseNet2D_up_block_concatNew(nn.Module):
    def __init__(self, skip_channels, input_channels, output_channels,
                 up_stride, dropout=False, prob=0):
        super().__init__()
        self.conv11 = nn.Conv2d(skip_channels + input_channels,
                                output_channels, kernel_size=(1, 1), padding=(0, 0))
        self.conv12 = nn.Conv2d(output_channels, output_channels,
                                kernel_size=(3, 3), padding=(1, 1))
        self.conv21 = nn.Conv2d(skip_channels + input_channels + output_channels,
                                output_channels, kernel_size=(1, 1), padding=(0, 0))
        self.conv22 = nn.Conv2d(output_channels, output_channels,
                                kernel_size=(3, 3), padding=(1, 1))
        self.relu = nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.slope = self.relu.negative_slope
        self._nw = 1

    def forward(self, prev_feature_map, x):
        s = self.up_stride
        x = x.contiguous()
        prev = prev_feature_map.contiguous()
        N, Cin, H, W = x.shape
        Hh, Ww = H * s, W * s
        Cskip = prev.shape[1]
        dev, dt = x.device, x.dtype
        Co = self.conv11.out_channels

        if s == 1:
            xu = x
        else:
            xu = torch.empty((N, Cin, Hh, Ww), device=dev, dtype=dt)
            upsample_kernel[(N * Cin, Hh)](x, xu, H, W, Hh, Ww, s,
                                           BLOCK_W=_np2(Ww), num_warps=4)

        BWc = _np2(Ww)
        C0 = Cin + Cskip
        nw = self._nw

        x1 = torch.empty((N, Co, Hh, Ww), device=dev, dtype=dt)
        fused_block_kernel[(N, Hh, Co)](xu, prev, prev,
                                    self.conv11.weight, self.conv11.bias,
                                    self.conv12.weight, self.conv12.bias,
                                    x1, Hh, Ww, self.slope,
                                    CA=Cin, CB=Cskip, CC=0, CIN=C0,
                                    CMID=Co, COUT=Co, BLOCK_W=BWc, num_warps=nw)

        out = torch.empty((N, Co, Hh, Ww), device=dev, dtype=dt)
        fused_block_kernel[(N, Hh, Co)](xu, prev, x1,
                                    self.conv21.weight, self.conv21.bias,
                                    self.conv22.weight, self.conv22.bias,
                                    out, Hh, Ww, self.slope,
                                    CA=Cin, CB=Cskip, CC=Co, CIN=C0 + Co,
                                    CMID=Co, COUT=Co, BLOCK_W=BWc, num_warps=nw)
        return out
