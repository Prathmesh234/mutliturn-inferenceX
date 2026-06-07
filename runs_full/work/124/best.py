import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr, w_ptr, b_ptr, g_ptr, beta_ptr, out_ptr,
                  stride_xa, Hin, Win, Wu, HW, M, s, eps,
                  BLOCK: tl.constexpr, CI: tl.constexpr):
    co = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < M
    b = offs // HW
    rem = offs % HW
    h = rem // Wu
    w = rem % Wu
    hin = h // s
    win = w // s
    base_x = b * (Hin * Win) + hin * Win + win
    acc = tl.zeros((BLOCK,), tl.float32)
    for a in range(CI):
        xval = tl.load(x_ptr + a * stride_xa + base_x, mask=mask, other=0.0)
        wval = tl.load(w_ptr + co * CI + a)
        acc += xval * wval
    acc += tl.load(b_ptr + co)
    acc = tl.where(mask, acc, 0.0)
    ssum = tl.sum(acc, axis=0)
    mean = ssum / M
    xc = tl.where(mask, acc - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / M
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(g_ptr + co)
    beta = tl.load(beta_ptr + co)
    y = (acc - mean) * rstd * g + beta
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + co * M + offs, y, mask=mask)


class UpsampleNew(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.trilinear = nn.Upsample(scale_factor=scale_factor)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.contiguous()
        Ci, D1, Hin, Win = x.shape
        sf = self.trilinear.scale_factor
        s = int(sf if not isinstance(sf, (tuple, list)) else sf[0])
        Hu = Hin * s
        Wu = Win * s
        Co = self.conv1.out_channels
        HW = Hu * Wu
        M = D1 * HW

        w = self.conv1.weight.reshape(Co, Ci).contiguous()
        bconv = self.conv1.bias.contiguous()
        out = torch.empty((Co, D1, Hu, Wu), device=x.device, dtype=x.dtype)

        BLOCK = triton.next_power_of_2(M)
        _fused_kernel[(Co,)](x, w, bconv, self.bn1.weight, self.bn1.bias, out,
                             D1 * Hin * Win, Hin, Win, Wu, HW, M, s,
                             float(self.bn1.eps),
                             BLOCK=BLOCK, CI=Ci, num_warps=4)
        return out
