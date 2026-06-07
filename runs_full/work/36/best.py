import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv_pool_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                      N, CIN, HIN, WIN, COUT, HP, WP, KH, KW,
                      BLOCK: tl.constexpr):
    pid0 = tl.program_id(0)   # n * COUT + co
    pid1 = tl.program_id(1)   # spatial block
    co = pid0 % COUT
    n = pid0 // COUT
    sp = pid1 * BLOCK + tl.arange(0, BLOCK)
    NSP = HP * WP
    mask = sp < NSP
    wp = sp % WP
    hp = sp // WP
    ho = hp * 2
    wo = wp * 2
    bv = tl.load(b_ptr + co)
    a00 = bv + 0.0 * sp
    a01 = a00
    a10 = a00
    a11 = a00
    w_co = co * CIN * KH * KW
    for ci in range(CIN):
        nbase = (n * CIN + ci) * HIN
        for kh in range(KH):
            for kw in range(KW):
                wv = tl.load(w_ptr + w_co + (ci * KH + kh) * KW + kw)
                i00 = (nbase + ho + kh) * WIN + wo + kw
                a00 += tl.load(x_ptr + i00, mask=mask, other=0.0) * wv
                a01 += tl.load(x_ptr + i00 + 1, mask=mask, other=0.0) * wv
                i10 = (nbase + ho + 1 + kh) * WIN + wo + kw
                a10 += tl.load(x_ptr + i10, mask=mask, other=0.0) * wv
                a11 += tl.load(x_ptr + i10 + 1, mask=mask, other=0.0) * wv
    m = tl.maximum(tl.maximum(a00, a01), tl.maximum(a10, a11))
    out_off = ((n * COUT + co) * HP) * WP + sp
    tl.store(out_ptr + out_off, m, mask=mask)


@triton.jit
def _mlp_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, out_ptr,
                K1, N1, K2, N2, K3, N3,
                BK1: tl.constexpr, BN1: tl.constexpr,
                BK2: tl.constexpr, BN2: tl.constexpr,
                BK3: tl.constexpr, BN3: tl.constexpr):
    m = tl.program_id(0)
    ok1 = tl.arange(0, BK1)
    xm = tl.load(x_ptr + m * K1 + ok1, mask=ok1 < K1, other=0.0)
    on1 = tl.arange(0, BN1)
    w1 = tl.load(w1_ptr + on1[:, None] * K1 + ok1[None, :],
                 mask=(on1[:, None] < N1) & (ok1[None, :] < K1), other=0.0)
    h1 = tl.sum(w1 * xm[None, :], axis=1) + tl.load(b1_ptr + on1, mask=on1 < N1, other=0.0)
    h1 = tl.where(on1 < N1, h1, 0.0)
    on2 = tl.arange(0, BN2)
    ok2 = tl.arange(0, BK2)
    w2 = tl.load(w2_ptr + on2[:, None] * K2 + ok2[None, :],
                 mask=(on2[:, None] < N2) & (ok2[None, :] < K2), other=0.0)
    h2 = tl.sum(w2 * h1[None, :], axis=1) + tl.load(b2_ptr + on2, mask=on2 < N2, other=0.0)
    h2 = tl.where(on2 < N2, h2, 0.0)
    on3 = tl.arange(0, BN3)
    ok3 = tl.arange(0, BK3)
    w3 = tl.load(w3_ptr + on3[:, None] * K3 + ok3[None, :],
                 mask=(on3[:, None] < N3) & (ok3[None, :] < K3), other=0.0)
    o = tl.sum(w3 * h2[None, :], axis=1) + tl.load(b3_ptr + on3, mask=on3 < N3, other=0.0)
    tl.store(out_ptr + m * N3 + on3, o, mask=on3 < N3)


def conv_pool(x, w, b):
    N, CIN, HIN, WIN = x.shape
    COUT, _, KH, KW = w.shape
    HP = (HIN - KH + 1) // 2
    WP = (WIN - KW + 1) // 2
    out = torch.empty((N, COUT, HP, WP), device=x.device, dtype=x.dtype)
    NSP = HP * WP
    BLOCK = triton.next_power_of_2(NSP)
    grid = (N * COUT, triton.cdiv(NSP, BLOCK))
    _conv_pool_kernel[grid](x, w, b, out, N, CIN, HIN, WIN, COUT, HP, WP, KH, KW,
                            BLOCK=BLOCK, num_warps=4)
    return out


class NetNew(nn.Module):
    def __init__(self):
        super(NetNew, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.contiguous()
        x = conv_pool(x, self.conv1.weight, self.conv1.bias)
        x = conv_pool(x, self.conv2.weight, self.conv2.bias)
        x = x.contiguous().view(-1, 16 * 5 * 5)
        M = x.shape[0]
        out = torch.empty((M, 10), device=x.device, dtype=x.dtype)
        _mlp_kernel[(M,)](x, self.fc1.weight, self.fc1.bias,
                          self.fc2.weight, self.fc2.bias,
                          self.fc3.weight, self.fc3.bias, out,
                          400, 120, 120, 84, 84, 10,
                          BK1=512, BN1=128, BK2=128, BN2=128, BK3=128, BN3=16,
                          num_warps=4)
        return out
