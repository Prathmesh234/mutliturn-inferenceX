import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _mfm_conv_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                     N, IC, H, W, OC, Hout, Wout,
                     KH: tl.constexpr, KW: tl.constexpr,
                     stride: tl.constexpr, pad: tl.constexpr,
                     BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    spatial = tl.arange(0, BLOCK)
    valid = spatial < (Hout * Wout)
    oh = spatial // Wout
    ow = spatial % Wout

    acc1 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK,), dtype=tl.float32)

    x_base = n * (IC * H * W)
    w_base1 = oc * (IC * KH * KW)
    w_base2 = (oc + OC) * (IC * KH * KW)

    for ic in range(IC):
        for kh in range(KH):
            ih = oh * stride - pad + kh
            for kw in range(KW):
                iw = ow * stride - pad + kw
                m = valid & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                inp = tl.load(x_ptr + x_base + ic * (H * W) + ih * W + iw,
                              mask=m, other=0.0)
                woff = ic * (KH * KW) + kh * KW + kw
                w1 = tl.load(w_ptr + w_base1 + woff)
                w2 = tl.load(w_ptr + w_base2 + woff)
                acc1 += inp * w1
                acc2 += inp * w2

    b1 = tl.load(b_ptr + oc)
    b2 = tl.load(b_ptr + oc + OC)
    acc1 += b1
    acc2 += b2
    out = tl.maximum(acc1, acc2)

    out_base = n * (OC * Hout * Wout) + oc * (Hout * Wout)
    tl.store(out_ptr + out_base + spatial, out, mask=valid)


@triton.jit
def _mfm_linear_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                       M, IC, OC,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, IC, BLOCK_K):
        kk = k + offs_k
        a = tl.load(x_ptr + offs_m[:, None] * IC + kk[None, :],
                    mask=(offs_m[:, None] < M) & (kk[None, :] < IC), other=0.0)
        # weight rows: oc (first half) and oc+OC (second half)
        w1 = tl.load(w_ptr + offs_n[:, None] * IC + kk[None, :],
                     mask=(offs_n[:, None] < OC) & (kk[None, :] < IC), other=0.0)
        w2 = tl.load(w_ptr + (offs_n[:, None] + OC) * IC + kk[None, :],
                     mask=(offs_n[:, None] < OC) & (kk[None, :] < IC), other=0.0)
        acc1 += tl.dot(a, tl.trans(w1))
        acc2 += tl.dot(a, tl.trans(w2))

    b1 = tl.load(b_ptr + offs_n, mask=offs_n < OC, other=0.0)
    b2 = tl.load(b_ptr + offs_n + OC, mask=offs_n < OC, other=0.0)
    acc1 += b1[None, :]
    acc2 += b2[None, :]
    out = tl.maximum(acc1, acc2)

    om = (offs_m[:, None] < M) & (offs_n[None, :] < OC)
    tl.store(out_ptr + offs_m[:, None] * OC + offs_n[None, :], out, mask=om)


class mfmNew(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, type=1):
        super(mfmNew, self).__init__()
        self.out_channels = out_channels
        self.type = type
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2 * out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)

    def forward(self, x):
        OC = self.out_channels
        if self.type == 1:
            x = x.contiguous()
            w = self.filter.weight.contiguous()
            b = self.filter.bias.contiguous()
            N, IC, H, W = x.shape
            KH, KW = w.shape[2], w.shape[3]
            stride = self.filter.stride[0]
            pad = self.filter.padding[0]
            Hout = (H + 2 * pad - KH) // stride + 1
            Wout = (W + 2 * pad - KW) // stride + 1
            out = torch.empty((N, OC, Hout, Wout), device=x.device, dtype=x.dtype)
            BLOCK = triton.next_power_of_2(Hout * Wout)
            grid = (N * OC,)
            _mfm_conv_kernel[grid](x, w, b, out, N, IC, H, W, OC, Hout, Wout,
                                   KH=KH, KW=KW, stride=stride, pad=pad,
                                   BLOCK=BLOCK, num_warps=2)
            return out
        else:
            w = self.filter.weight.contiguous()
            b = self.filter.bias.contiguous()
            IC = w.shape[1]
            orig_shape = x.shape
            x2 = x.reshape(-1, IC).contiguous()
            M = x2.shape[0]
            out = torch.empty((M, OC), device=x.device, dtype=x.dtype)
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(OC, BLOCK_N))
            _mfm_linear_kernel[grid](x2, w, b, out, M, IC, OC,
                                     BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                                     BLOCK_K=BLOCK_K, num_warps=4)
            return out.reshape(*orig_shape[:-1], OC)
