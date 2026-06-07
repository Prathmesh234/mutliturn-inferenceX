import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr, w_ptr, out_ptr,
                  N, Cin, GR, H, W, Tout,
                  K: tl.constexpr, PAD: tl.constexpr, BLOCK: tl.constexpr):
    pid_nc = tl.program_id(0)
    pid_s = tl.program_id(1)
    n = pid_nc // GR
    oc = pid_nc % GR
    offs = pid_s * BLOCK + tl.arange(0, BLOCK)
    HW = H * W
    mask_s = offs < HW
    oh = offs // W
    ow = offs % W
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    x_base = n * Cin * HW
    w_base = oc * Cin * K * K
    for ic in range(Cin):
        for kh in range(K):
            ih = oh + kh - PAD
            for kw in range(K):
                iw = ow + kw - PAD
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                xoff = x_base + ic * HW + ih * W + iw
                xv = tl.load(x_ptr + xoff, mask=mask_s & in_bounds, other=0.0)
                wv = tl.load(w_ptr + w_base + ic * K * K + kh * K + kw)
                acc += xv * wv
    v = tl.maximum(acc, 0.0)
    xc = tl.load(x_ptr + x_base + oc * HW + offs, mask=mask_s, other=0.0)
    out_n = n * Tout * HW
    tl.store(out_ptr + out_n + oc * HW + offs, xc, mask=mask_s)
    tl.store(out_ptr + out_n + (GR + oc) * HW + offs, xc + v, mask=mask_s)
    tl.store(out_ptr + out_n + (2 * GR + oc) * HW + offs, v, mask=mask_s)


@triton.jit
def _conv_relu_kernel(x_ptr, w_ptr, out_ptr, N, Cin, Cout, H, W,
                      K: tl.constexpr, PAD: tl.constexpr, BLOCK: tl.constexpr):
    pid_nc = tl.program_id(0)
    pid_s = tl.program_id(1)
    n = pid_nc // Cout
    oc = pid_nc % Cout
    offs = pid_s * BLOCK + tl.arange(0, BLOCK)
    HW = H * W
    mask_s = offs < HW
    oh = offs // W
    ow = offs % W
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    x_base = n * Cin * HW
    w_base = oc * Cin * K * K
    for ic in range(Cin):
        for kh in range(K):
            ih = oh + kh - PAD
            for kw in range(K):
                iw = ow + kw - PAD
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                xoff = x_base + ic * HW + ih * W + iw
                xv = tl.load(x_ptr + xoff, mask=mask_s & in_bounds, other=0.0)
                wv = tl.load(w_ptr + w_base + ic * K * K + kh * K + kw)
                acc += xv * wv
    acc = tl.maximum(acc, 0.0)
    tl.store(out_ptr + (n * Cout + oc) * HW + offs, acc, mask=mask_s)


@triton.jit
def _add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a + b, mask=mask)


def _conv_relu(x, weight):
    N, Cin, H, W = x.shape
    Cout = weight.shape[0]
    K = weight.shape[2]
    PAD = (K - 1) // 2
    out = torch.empty((N, Cout, H, W), device=x.device, dtype=x.dtype)
    BLOCK = 256
    grid = (N * Cout, triton.cdiv(H * W, BLOCK))
    _conv_relu_kernel[grid](x, weight, out, N, Cin, Cout, H, W,
                            K=K, PAD=PAD, BLOCK=BLOCK, num_warps=4)
    return out


def _add(a, b):
    a = a.contiguous()
    b = b.contiguous()
    out = torch.empty_like(a)
    n = a.numel()
    BLOCK = 1024
    _add_kernel[(triton.cdiv(n, BLOCK),)](a, b, out, n, BLOCK=BLOCK, num_warps=4)
    return out


class make_residual_dense_ver2New(nn.Module):

    def __init__(self, nChannels, nChannels_, growthRate, kernel_size=3):
        super(make_residual_dense_ver2New, self).__init__()
        if nChannels == nChannels_:
            self.conv = nn.Conv2d(nChannels_, growthRate, kernel_size=
                kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        else:
            self.conv = nn.Conv2d(nChannels_ + growthRate, growthRate,
                kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                bias=False)
        self.nChannels_ = nChannels_
        self.nChannels = nChannels
        self.growthrate = growthRate

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv.weight.contiguous()
        N, Cin, H, W = x.shape
        K = weight.shape[2]
        PAD = (K - 1) // 2
        GR = self.growthrate
        if Cin == self.nChannels and self.nChannels == GR:
            Tout = self.nChannels + 2 * GR
            out = torch.empty((N, Tout, H, W), device=x.device, dtype=x.dtype)
            HW = H * W
            BLOCK = triton.next_power_of_2(HW)
            grid = (N * GR, 1)
            _fused_kernel[grid](x, weight, out, N, Cin, GR, H, W, Tout,
                                K=K, PAD=PAD, BLOCK=BLOCK, num_warps=1)
            return out
        outoflayer = _conv_relu(x, weight)
        if x.shape[1] == self.nChannels:
            out = torch.cat((x, _add(x, outoflayer)), 1)
        else:
            out = torch.cat((x[:, :self.nChannels, :, :],
                             _add(x[:, self.nChannels:self.nChannels +
                                  self.growthrate, :, :], outoflayer),
                             x[:, self.nChannels + self.growthrate:, :, :]), 1)
        out = torch.cat((out, outoflayer), 1)
        return out
