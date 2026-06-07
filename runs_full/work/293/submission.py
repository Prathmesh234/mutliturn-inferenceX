import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_pool_pos(x_ptr, w_ptr, b_ptr, y_ptr,
                  Cin, H, W, Cout, PH, PW,
                  BLOCK_C: tl.constexpr, CIN: tl.constexpr,
                  KH: tl.constexpr, KW: tl.constexpr):
    pid = tl.program_id(0)
    npos = PH * PW
    n = pid // npos
    rem = pid % npos
    ph = rem // PW
    pw = rem % PW
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < Cout
    maxval = tl.full((BLOCK_C,), -1e30, tl.float32)
    for dh in range(0, 2):
        for dw in range(0, 2):
            oh = ph * 2 + dh
            ow = pw * 2 + dw
            acc = tl.zeros((BLOCK_C,), tl.float32)
            for ci in range(0, CIN):
                for kh in range(0, KH):
                    for kw in range(0, KW):
                        xval = tl.load(x_ptr + n * Cin * H * W + ci * H * W + (oh + kh) * W + (ow + kw))
                        wvec = tl.load(w_ptr + offs_c * (Cin * KH * KW) + ci * KH * KW + kh * KW + kw,
                                       mask=mask_c, other=0.0)
                        acc += xval * wvec
            maxval = tl.maximum(maxval, acc)
    bvec = tl.load(b_ptr + offs_c, mask=mask_c, other=0.0)
    out = tl.maximum(maxval + bvec, 0.0)
    yp = y_ptr + n * Cout * PH * PW + offs_c * (PH * PW) + ph * PW + pw
    tl.store(yp, out, mask=mask_c)


@triton.jit
def conv_pool_batch(x_ptr, w_ptr, b_ptr, y_ptr,
                    Cin, H, W, Cout, PH, PW,
                    BN: tl.constexpr, BLOCK_C: tl.constexpr, CIN: tl.constexpr,
                    KH: tl.constexpr, KW: tl.constexpr, NB: tl.constexpr):
    pid = tl.program_id(0)
    ph = pid // PW
    pw = pid % PW
    offs_c = tl.arange(0, BLOCK_C)[None, :]
    offs_n = tl.arange(0, BN)[:, None]
    mask = (offs_n < NB) & (offs_c < Cout)
    maxval = tl.full((BN, BLOCK_C), -1e30, tl.float32)
    for dh in range(0, 2):
        for dw in range(0, 2):
            oh = ph * 2 + dh
            ow = pw * 2 + dw
            acc = tl.zeros((BN, BLOCK_C), tl.float32)
            for ci in range(0, CIN):
                for kh in range(0, KH):
                    for kw in range(0, KW):
                        xv = tl.load(x_ptr + offs_n * (Cin * H * W) + ci * H * W + (oh + kh) * W + (ow + kw),
                                     mask=offs_n < NB, other=0.0)
                        wv = tl.load(w_ptr + offs_c * (Cin * KH * KW) + ci * KH * KW + kh * KW + kw,
                                     mask=offs_c < Cout, other=0.0)
                        acc += xv * wv
            maxval = tl.maximum(maxval, acc)
    bv = tl.load(b_ptr + offs_c, mask=offs_c < Cout, other=0.0)
    out = tl.maximum(maxval + bv, 0.0)
    yp = y_ptr + offs_n * (Cout * PH * PW) + offs_c * (PH * PW) + ph * PW + pw
    tl.store(yp, out, mask=mask)


@triton.jit
def conv2_vec(x_ptr, w_ptr, b_ptr, y_ptr,
              Cin, H, W, Cout, PH, PW,
              BLOCK_C: tl.constexpr, CIN: tl.constexpr,
              KH: tl.constexpr, KW: tl.constexpr, KWP: tl.constexpr):
    pid = tl.program_id(0)
    npos = PH * PW
    n = pid // npos
    rem = pid % npos
    ph = rem // PW
    pw = rem % PW
    offs_c = tl.arange(0, BLOCK_C)[:, None]
    mask_c = offs_c < Cout
    kwr = tl.arange(0, KWP)[None, :]
    mkw = kwr < KW
    maxval = tl.full((BLOCK_C, 1), -1e30, tl.float32)
    for dh in range(0, 2):
        for dw in range(0, 2):
            oh = ph * 2 + dh
            ow = pw * 2 + dw
            acc = tl.zeros((BLOCK_C, 1), tl.float32)
            for ci in range(0, CIN):
                for kh in range(0, KH):
                    ih = oh + kh
                    xrow = tl.load(x_ptr + n * Cin * H * W + ci * H * W + ih * W + ow + kwr,
                                   mask=mkw, other=0.0)
                    wmat = tl.load(w_ptr + offs_c * (Cin * KH * KW) + ci * KH * KW + kh * KW + kwr,
                                   mask=mask_c & mkw, other=0.0)
                    acc += tl.sum(xrow * wmat, axis=1)[:, None]
            maxval = tl.maximum(maxval, acc)
    bv = tl.load(b_ptr + offs_c, mask=mask_c, other=0.0)
    out = tl.maximum(maxval + bv, 0.0)
    yp = y_ptr + n * Cout * PH * PW + offs_c * (PH * PW) + ph * PW + pw
    tl.store(yp, out, mask=mask_c)


@triton.jit
def mlp_kernel(x_ptr, w1, b1, w2, b2, w3, b3, y_ptr):
    offs_m = tl.arange(0, 16)
    offs_n = tl.arange(0, 128)
    h1 = tl.zeros((16, 128), tl.float32)
    for k0 in range(0, 400, 128):
        offs_k = k0 + tl.arange(0, 128)
        a = tl.load(x_ptr + offs_m[:, None] * 400 + offs_k[None, :],
                    mask=(offs_m[:, None] < 4) & (offs_k[None, :] < 400), other=0.0)
        w = tl.load(w1 + offs_n[None, :] * 400 + offs_k[:, None],
                    mask=(offs_n[None, :] < 120) & (offs_k[:, None] < 400), other=0.0)
        h1 += tl.dot(a, w)
    bb = tl.load(b1 + offs_n, mask=offs_n < 120, other=0.0)
    h1 = tl.maximum(h1 + bb[None, :], 0.0)
    offs_k2 = tl.arange(0, 128)
    w = tl.load(w2 + offs_n[None, :] * 120 + offs_k2[:, None],
                mask=(offs_n[None, :] < 84) & (offs_k2[:, None] < 120), other=0.0)
    h2 = tl.dot(h1, w)
    bb = tl.load(b2 + offs_n, mask=offs_n < 84, other=0.0)
    h2 = tl.maximum(h2 + bb[None, :], 0.0)
    offs_n3 = tl.arange(0, 16)
    w = tl.load(w3 + offs_n3[None, :] * 84 + offs_k2[:, None],
                mask=(offs_n3[None, :] < 10) & (offs_k2[:, None] < 84), other=0.0)
    out = tl.dot(h2, w)
    bb = tl.load(b3 + offs_n3, mask=offs_n3 < 10, other=0.0)
    out = out + bb[None, :]
    tl.store(y_ptr + offs_m[:, None] * 10 + offs_n3[None, :], out,
             mask=(offs_m[:, None] < 4) & (offs_n3[None, :] < 10))


def _conv_batch(x, weight, bias):
    N, Cin, H, W = x.shape
    Cout, _, KH, KW = weight.shape
    PH = (H - KH + 1) // 2
    PW = (W - KW + 1) // 2
    y = torch.empty((N, Cout, PH, PW), device=x.device, dtype=x.dtype)
    conv_pool_batch[(PH * PW,)](x, weight, bias, y, Cin, H, W, Cout, PH, PW,
                                BN=triton.next_power_of_2(N), BLOCK_C=triton.next_power_of_2(Cout),
                                CIN=Cin, KH=KH, KW=KW, NB=N, num_warps=1)
    return y


def _conv_pos(x, weight, bias):
    N, Cin, H, W = x.shape
    Cout, _, KH, KW = weight.shape
    PH = (H - KH + 1) // 2
    PW = (W - KW + 1) // 2
    y = torch.empty((N, Cout, PH, PW), device=x.device, dtype=x.dtype)
    conv2_vec[(N * PH * PW,)](x, weight, bias, y, Cin, H, W, Cout, PH, PW,
                              BLOCK_C=triton.next_power_of_2(Cout),
                              CIN=Cin, KH=KH, KW=KW, KWP=8, num_warps=4)
    return y


class SimplenetNew(nn.Module):
    def __init__(self):
        super(SimplenetNew, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.contiguous()
        x = _conv_batch(x, self.conv1.weight, self.conv1.bias)
        x = _conv_pos(x, self.conv2.weight, self.conv2.bias)
        x = x.view(-1, 16 * 5 * 5)
        out = torch.empty((x.shape[0], 10), device=x.device, dtype=x.dtype)
        mlp_kernel[(1,)](x, self.fc1.weight, self.fc1.bias,
                         self.fc2.weight, self.fc2.bias,
                         self.fc3.weight, self.fc3.bias, out, num_warps=4)
        return out
