import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv_kernel(x_ptr, w_ptr, b_ptr, scale_ptr, shift_ptr, out_ptr,
                 N, Cin, H, W, Cout, OH, OW, KH, KW, stride, pad,
                 HAS_BN: tl.constexpr, HAS_RELU: tl.constexpr,
                 BLOCK_OC: tl.constexpr):
    pid_pos = tl.program_id(0)
    pid_oc = tl.program_id(1)

    ow = pid_pos % OW
    tmp = pid_pos // OW
    oh = tmp % OH
    n = tmp // OH

    oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = oc < Cout

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    ih0 = oh * stride - pad
    iw0 = ow * stride - pad

    for cin in range(Cin):
        for kh in range(KH):
            ih = ih0 + kh
            for kw in range(KW):
                iw = iw0 + kw
                valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                x_off = n * Cin * H * W + cin * H * W + ih * W + iw
                xval = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                w_off = oc * (Cin * KH * KW) + cin * KH * KW + kh * KW + kw
                wval = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)
                acc += wval * xval

    bval = tl.load(b_ptr + oc, mask=mask_oc, other=0.0)
    acc += bval

    if HAS_BN:
        scale = tl.load(scale_ptr + oc, mask=mask_oc, other=0.0)
        shift = tl.load(shift_ptr + oc, mask=mask_oc, other=0.0)
        acc = acc * scale + shift

    if HAS_RELU:
        acc = tl.maximum(acc, 0.0)

    out_off = n * Cout * OH * OW + oc * OH * OW + oh * OW + ow
    tl.store(out_ptr + out_off, acc, mask=mask_oc)


class Conv2dNew(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, same_padding=False, bn=False):
        super(Conv2dNew, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0,
                                 affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight
        b = self.conv.bias
        N, Cin, H, W = x.shape
        Cout, _, KH, KW = w.shape
        stride = self.conv.stride[0]
        pad = self.conv.padding[0]
        OH = (H + 2 * pad - KH) // stride + 1
        OW = (W + 2 * pad - KW) // stride + 1

        out = torch.empty((N, Cout, OH, OW), device=x.device, dtype=x.dtype)

        HAS_BN = self.bn is not None
        if HAS_BN:
            eps = self.bn.eps
            inv = self.bn.weight / torch.sqrt(self.bn.running_var + eps)
            scale = inv.contiguous()
            shift = (self.bn.bias - self.bn.running_mean * inv).contiguous()
        else:
            scale = torch.empty(1, device=x.device, dtype=x.dtype)
            shift = scale

        BLOCK_OC = max(16, triton.next_power_of_2(Cout))
        grid = (N * OH * OW, triton.cdiv(Cout, BLOCK_OC))
        _conv_kernel[grid](x, w, b, scale, shift, out,
                           N, Cin, H, W, Cout, OH, OW, KH, KW, stride, pad,
                           HAS_BN=HAS_BN, HAS_RELU=self.relu is not None,
                           BLOCK_OC=BLOCK_OC, num_warps=1)
        return out
