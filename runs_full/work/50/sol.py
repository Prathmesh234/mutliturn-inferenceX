import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _mfm_conv_kernel(x_ptr, w_ptr, b_ptr, res_ptr, out_ptr,
                     N,
                     IC: tl.constexpr, OC: tl.constexpr,
                     H: tl.constexpr, W: tl.constexpr,
                     add_res: tl.constexpr,
                     BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * OC * H * W
    mask = offs < total

    ow = offs % W
    oh = (offs // W) % H
    oc = (offs // (W * H)) % OC
    n = offs // (W * H * OC)

    acc_a = tl.load(b_ptr + oc, mask=mask, other=0.0)
    acc_b = tl.load(b_ptr + (oc + OC), mask=mask, other=0.0)

    for ci in range(IC):
        for kh in range(3):
            ih = oh + kh - 1
            vy = (ih >= 0) & (ih < H)
            for kw in range(3):
                iw = ow + kw - 1
                vx = (iw >= 0) & (iw < W)
                valid = mask & vy & vx
                xoff = ((n * IC + ci) * H + ih) * W + iw
                xoff = tl.where(valid, xoff, 0)
                xv = tl.load(x_ptr + xoff, mask=valid, other=0.0)
                woff_a = ((oc * IC + ci) * 3 + kh) * 3 + kw
                woff_b = (((oc + OC) * IC + ci) * 3 + kh) * 3 + kw
                wa = tl.load(w_ptr + woff_a, mask=mask, other=0.0)
                wb = tl.load(w_ptr + woff_b, mask=mask, other=0.0)
                acc_a += xv * wa
                acc_b += xv * wb

    res = tl.maximum(acc_a, acc_b)
    if add_res:
        r = tl.load(res_ptr + offs, mask=mask, other=0.0)
        res = res + r
    tl.store(out_ptr + offs, res, mask=mask)


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

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


def _run_mfm(filt, x, out, res, add_res):
    N, IC, H, W = x.shape
    OC = filt.out_channels // 2  # Conv2d produces 2*out_channels
    w = filt.weight.contiguous()
    b = filt.bias.contiguous()
    total = N * OC * H * W
    BLOCK = 128
    grid = (triton.cdiv(total, BLOCK),)
    res_arg = res if res is not None else x
    _mfm_conv_kernel[grid](x, w, b, res_arg, out, N, IC, OC, H, W,
                           add_res, BLOCK=BLOCK, num_warps=1)
    return out


class resblockNew(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(resblockNew, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1,
            padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1,
            padding=1)

    def forward(self, x):
        x = x.contiguous()
        N, IC, H, W = x.shape
        OC = self.conv1.out_channels
        out1 = torch.empty((N, OC, H, W), device=x.device, dtype=x.dtype)
        _run_mfm(self.conv1.filter, x, out1, None, False)
        # conv2 takes out1, adds residual x
        out2 = torch.empty((N, OC, H, W), device=x.device, dtype=x.dtype)
        _run_mfm(self.conv2.filter, out1, out2, x, True)
        return out2
