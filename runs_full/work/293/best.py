import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_pool_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
                     Cin, H, W, Cout, OH, OW, PH, PW,
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
                        ih = oh + kh
                        iw = ow + kw
                        xval = tl.load(x_ptr + n * Cin * H * W + ci * H * W + ih * W + iw)
                        wvec = tl.load(w_ptr + offs_c * (Cin * KH * KW) + ci * KH * KW + kh * KW + kw,
                                       mask=mask_c, other=0.0)
                        acc += xval * wvec
            maxval = tl.maximum(maxval, acc)

    bvec = tl.load(b_ptr + offs_c, mask=mask_c, other=0.0)
    out = tl.maximum(maxval + bvec, 0.0)
    y_ptrs = y_ptr + n * Cout * PH * PW + offs_c * (PH * PW) + ph * PW + pw
    tl.store(y_ptrs, out, mask=mask_c)


@triton.jit
def mlp_kernel(x_ptr, w1, b1, w2, b2, w3, b3, y_ptr):
    offs_m = tl.arange(0, 16)
    offs_n = tl.arange(0, 128)
    # ---- fc1: [4,400] x [120,400]^T -> [4,120], relu ----
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
    # ---- fc2: [4,120] x [84,120]^T -> [4,84], relu ----
    offs_k2 = tl.arange(0, 128)
    w = tl.load(w2 + offs_n[None, :] * 120 + offs_k2[:, None],
                mask=(offs_n[None, :] < 84) & (offs_k2[:, None] < 120), other=0.0)
    h2 = tl.dot(h1, w)
    bb = tl.load(b2 + offs_n, mask=offs_n < 84, other=0.0)
    h2 = tl.maximum(h2 + bb[None, :], 0.0)
    # ---- fc3: [4,84] x [10,84]^T -> [4,10] ----
    offs_n3 = tl.arange(0, 16)
    w = tl.load(w3 + offs_n3[None, :] * 84 + offs_k2[:, None],
                mask=(offs_n3[None, :] < 10) & (offs_k2[:, None] < 84), other=0.0)
    out = tl.dot(h2, w)
    bb = tl.load(b3 + offs_n3, mask=offs_n3 < 10, other=0.0)
    out = out + bb[None, :]
    tl.store(y_ptr + offs_m[:, None] * 10 + offs_n3[None, :], out,
             mask=(offs_m[:, None] < 4) & (offs_n3[None, :] < 10))


def _conv_pool(x, weight, bias):
    N, Cin, H, W = x.shape
    Cout, _, KH, KW = weight.shape
    OH = H - KH + 1
    OW = W - KW + 1
    PH = OH // 2
    PW = OW // 2
    y = torch.empty((N, Cout, PH, PW), device=x.device, dtype=x.dtype)
    BLOCK_C = triton.next_power_of_2(Cout)
    grid = (N * PH * PW,)
    conv_pool_kernel[grid](x, weight, bias, y,
                           Cin, H, W, Cout, OH, OW, PH, PW,
                           BLOCK_C=BLOCK_C, CIN=Cin, KH=KH, KW=KW, num_warps=2)
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
        x = _conv_pool(x, self.conv1.weight, self.conv1.bias)
        x = _conv_pool(x, self.conv2.weight, self.conv2.bias)
        x = x.view(-1, 16 * 5 * 5)
        out = torch.empty((x.shape[0], 10), device=x.device, dtype=x.dtype)
        mlp_kernel[(1,)](x, self.fc1.weight, self.fc1.bias,
                         self.fc2.weight, self.fc2.bias,
                         self.fc3.weight, self.fc3.bias, out, num_warps=4)
        return out
