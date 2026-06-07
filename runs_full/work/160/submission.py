import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _scse_kernel(x_ptr, wsq_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
                 C, Cr, HW,
                 BLOCK_C: tl.constexpr, BLOCK_CR: tl.constexpr, BLOCK_HW: tl.constexpr):
    n = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    offs_hw = tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW
    offs_cr = tl.arange(0, BLOCK_CR)
    mask_cr = offs_cr < Cr

    # load x[n] as [C, HW]
    xptrs = x_ptr + n * C * HW + offs_c[:, None] * HW + offs_hw[None, :]
    xmask = mask_c[:, None] & mask_hw[None, :]
    xv = tl.load(xptrs, mask=xmask, other=0.0)

    # spatial attention: zs[hw] = sigmoid(sum_c x[c,hw]*wsq[c])
    wsq = tl.load(wsq_ptr + offs_c, mask=mask_c, other=0.0)
    zs = tl.sigmoid(tl.sum(xv * wsq[:, None], axis=0))  # [HW]

    # channel attention
    pooled = tl.sum(xv, axis=1) / HW  # [C]
    w1ptrs = w1_ptr + offs_cr[:, None] * C + offs_c[None, :]
    w1v = tl.load(w1ptrs, mask=mask_cr[:, None] & mask_c[None, :], other=0.0)
    h1 = tl.sum(w1v * pooled[None, :], axis=1) + tl.load(b1_ptr + offs_cr, mask=mask_cr, other=0.0)
    h1 = tl.maximum(h1, 0.0)
    w2ptrs = w2_ptr + offs_c[:, None] * Cr + offs_cr[None, :]
    w2v = tl.load(w2ptrs, mask=mask_c[:, None] & mask_cr[None, :], other=0.0)
    zc = tl.sigmoid(tl.sum(w2v * h1[None, :], axis=1) + tl.load(b2_ptr + offs_c, mask=mask_c, other=0.0))  # [C]

    out = xv * zs[None, :] + xv * zc[:, None]
    tl.store(out_ptr + n * C * HW + offs_c[:, None] * HW + offs_hw[None, :], out, mask=xmask)


def _next_pow2(x):
    return 1 << (max(x, 1) - 1).bit_length()


class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z


class SCseNew(nn.Module):
    def __init__(self, dim):
        super(SCseNew, self).__init__()
        self.satt = SpatialAttention2d(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        x = x.contiguous()
        N, C, H, W = x.shape
        HW = H * W
        Cr = self.catt.conv1.weight.shape[0]

        w_sq = self.satt.squeeze.weight.reshape(C).contiguous()
        w1 = self.catt.conv1.weight.reshape(Cr, C).contiguous()
        b1 = self.catt.conv1.bias.contiguous()
        w2 = self.catt.conv2.weight.reshape(C, Cr).contiguous()
        b2 = self.catt.conv2.bias.contiguous()

        out = torch.empty_like(x)
        BLOCK_C = _next_pow2(C)
        BLOCK_CR = _next_pow2(Cr)
        BLOCK_HW = _next_pow2(HW)
        _scse_kernel[(N,)](x, w_sq, w1, b1, w2, b2, out, C, Cr, HW,
                           BLOCK_C=BLOCK_C, BLOCK_CR=BLOCK_CR, BLOCK_HW=BLOCK_HW, num_warps=2)
        return out
