import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                  N, CIN, H, W, COUT, OH, OW,
                  KH: tl.constexpr, KW: tl.constexpr, stride: tl.constexpr,
                  HAS_BIAS: tl.constexpr, RELU: tl.constexpr,
                  BLOCK_OW: tl.constexpr):
    pid = tl.program_id(0)
    oh = pid % OH
    tmp = pid // OH
    co = tmp % COUT
    n = tmp // COUT
    ow = tl.arange(0, BLOCK_OW)
    mask_ow = ow < OW
    acc = tl.zeros([BLOCK_OW], dtype=tl.float32)
    for ci in range(CIN):
        for kh in range(KH):
            ih = oh * stride + kh
            for kw in range(KW):
                iw = ow * stride + kw
                in_off = ((n * CIN + ci) * H + ih) * W + iw
                xval = tl.load(x_ptr + in_off, mask=mask_ow, other=0.0)
                wval = tl.load(w_ptr + ((co * CIN + ci) * KH + kh) * KW + kw)
                acc += xval * wval
    if HAS_BIAS:
        acc += tl.load(b_ptr + co)
    if RELU:
        acc = tl.maximum(acc, 0.0)
    out_off = ((n * COUT + co) * OH + oh) * OW + ow
    tl.store(out_ptr + out_off, acc, mask=mask_ow)


@triton.jit
def linear_kernel(x_ptr, w_ptr, b_ptr, out_ptr, M, K, Nout,
                  RELU: tl.constexpr, SIGMOID: tl.constexpr,
                  BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_off < Nout
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < K
        x = tl.load(x_ptr + pid_m * K + k, mask=mask_k, other=0.0)
        w = tl.load(w_ptr + n_off[:, None] * K + k[None, :],
                    mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc += tl.sum(w * x[None, :], axis=1)
    acc += tl.load(b_ptr + n_off, mask=mask_n, other=0.0)
    if RELU:
        acc = tl.maximum(acc, 0.0)
    if SIGMOID:
        acc = 1.0 / (1.0 + tl.exp(-acc))
    tl.store(out_ptr + pid_m * Nout + n_off, acc, mask=mask_n)


def _conv(x, weight, bias, stride, relu):
    N, CIN, H, W = x.shape
    COUT, _, KH, KW = weight.shape
    OH = (H - KH) // stride + 1
    OW = (W - KW) // stride + 1
    out = torch.empty((N, COUT, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_OW = triton.next_power_of_2(OW)
    grid = (N * COUT * OH,)
    b_ptr = bias if bias is not None else x
    nw = 1 if BLOCK_OW <= 32 else 4
    conv2d_kernel[grid](x, weight, b_ptr, out,
                        N, CIN, H, W, COUT, OH, OW,
                        KH=KH, KW=KW, stride=stride,
                        HAS_BIAS=bias is not None, RELU=relu,
                        BLOCK_OW=BLOCK_OW, num_warps=nw)
    return out


@triton.jit
def fused_fc_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
                    K, N1: tl.constexpr, BLOCK_K: tl.constexpr):
    m = tl.program_id(0)
    n = tl.arange(0, N1)
    acc = tl.zeros([N1], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < K
        xk = tl.load(x_ptr + m * K + k, mask=mask_k, other=0.0)
        w1 = tl.load(w1_ptr + n[:, None] * K + k[None, :],
                     mask=mask_k[None, :], other=0.0)
        acc += tl.sum(w1 * xk[None, :], axis=1)
    h = tl.maximum(acc + tl.load(b1_ptr + n), 0.0)
    w2 = tl.load(w2_ptr + n)
    o = tl.sum(h * w2) + tl.load(b2_ptr)
    o = 1.0 / (1.0 + tl.exp(-o))
    tl.store(out_ptr + m, o)


def _fused_fc(x, w1, b1, w2, b2):
    M, K = x.shape
    N1 = w1.shape[0]
    out = torch.empty((M, 1), device=x.device, dtype=x.dtype)
    fused_fc_kernel[(M,)](x, w1, b1, w2, b2, out, K,
                          N1=N1, BLOCK_K=64, num_warps=4)
    return out


def _linear(x, weight, bias, relu, sigmoid):
    M, K = x.shape
    Nout = weight.shape[0]
    out = torch.empty((M, Nout), device=x.device, dtype=x.dtype)
    BLOCK_N = 64
    BLOCK_K = 256
    grid = (M, triton.cdiv(Nout, BLOCK_N))
    linear_kernel[grid](x, weight, bias, out, M, K, Nout,
                        RELU=relu, SIGMOID=sigmoid,
                        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=2)
    return out


class PolicyNew(nn.Module):

    def __init__(self):
        super(PolicyNew, self).__init__()
        self.conv1 = nn.Conv2d(2, 4, kernel_size=6, stride=2, bias=False)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
        self.size = 9 * 9 * 16
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.contiguous()
        x = _conv(x, self.conv1.weight, None, 2, relu=True)
        x = _conv(x, self.conv2.weight, self.conv2.bias, 4, relu=True)
        x = x.contiguous().view(-1, self.size)
        x = _fused_fc(x, self.fc1.weight, self.fc1.bias,
                      self.fc2.weight, self.fc2.bias)
        return x
