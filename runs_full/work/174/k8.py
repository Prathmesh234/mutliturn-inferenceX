import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr, w0_ptr, b0_ptr, p_ptr, w1_ptr, b1_ptr, out_ptr,
                  N, H, W,
                  IC: tl.constexpr, OC: tl.constexpr,
                  ICg: tl.constexpr, OCg: tl.constexpr,
                  KH: tl.constexpr, KW: tl.constexpr, PAD: tl.constexpr,
                  IC1: tl.constexpr, OC1: tl.constexpr,
                  BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * H * W
    mask = offs < total

    w = offs % W
    h = (offs // W) % H
    n = offs // (W * H)

    alpha = tl.load(p_ptr)

    for oc1 in tl.static_range(OC1):
        o = tl.load(b1_ptr + oc1) + tl.zeros([BLOCK], tl.float32)
        for mc in tl.static_range(IC1):
            group = mc // OCg
            ic_base = group * ICg
            acc = tl.load(b0_ptr + mc) + tl.zeros([BLOCK], tl.float32)
            for icl in tl.static_range(ICg):
                ic = ic_base + icl
                for kh in tl.static_range(KH):
                    for kw in tl.static_range(KW):
                        h_in = h + kh - PAD
                        w_in = w + kw - PAD
                        vmask = mask & (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                        x_off = n * (IC * H * W) + ic * (H * W) + h_in * W + w_in
                        xv = tl.load(x_ptr + x_off, mask=vmask, other=0.0)
                        wv = tl.load(w0_ptr + mc * (ICg * KH * KW) + icl * (KH * KW) + kh * KW + kw)
                        acc += xv * wv
            acc = tl.where(acc >= 0, acc, alpha * acc)
            wv1 = tl.load(w1_ptr + oc1 * IC1 + mc)
            o += wv1 * acc
        out_off = n * (OC1 * H * W) + oc1 * (H * W) + h * W + w
        tl.store(out_ptr + out_off, o, mask=mask)


class GblockNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super(GblockNew, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            padding=1, groups=groups)
        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
            padding=0)

    def forward(self, x):
        x = x.contiguous()
        N, IC, H, W = x.shape
        OC = self.conv0.out_channels
        groups = self.conv0.groups
        ICg = IC // groups
        OCg = OC // groups
        IC1 = self.conv1.in_channels
        OC1 = self.conv1.out_channels

        out = torch.empty((N, OC1, H, W), device=x.device, dtype=x.dtype)
        total = N * H * W
        BLOCK = 8
        grid = (triton.cdiv(total, BLOCK),)
        _fused_kernel[grid](x, self.conv0.weight, self.conv0.bias,
                            self.relu.weight, self.conv1.weight, self.conv1.bias,
                            out, N, H, W,
                            IC=IC, OC=OC, ICg=ICg, OCg=OCg,
                            KH=3, KW=3, PAD=1, IC1=IC1, OC1=OC1,
                            BLOCK=BLOCK, num_warps=1)
        return out
