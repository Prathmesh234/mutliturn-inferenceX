import math
import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    scale, lr_mul, neg_slope, act_scale,
    HAS_BIAS: tl.constexpr, ACT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mmask = offs_m < M
    nmask = offs_n < N
    kmask = offs_k < K

    # x: [BLOCK_M, BLOCK_K], w: [BLOCK_N, BLOCK_K]
    x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=mmask[:, None] & kmask[None, :], other=0.0)
    w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :],
                mask=nmask[:, None] & kmask[None, :], other=0.0)

    # acc[m,n] = sum_k x[m,k]*w[n,k]
    acc = tl.sum(x[:, None, :] * w[None, :, :], axis=2)
    acc = acc * scale
    if HAS_BIAS:
        b = tl.load(b_ptr + offs_n, mask=nmask, other=0.0)
        acc += (b * lr_mul)[None, :]
    if ACT:
        acc = tl.where(acc >= 0, acc, acc * neg_slope) * act_scale

    out_off = offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptr + out_off, acc, mask=mmask[:, None] & nmask[None, :])


class EqualLinearNew(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1,
                 activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.activation = activation
        self.scale = 1 / math.sqrt(in_dim) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        out_dim, in_dim = self.weight.shape
        x = input.reshape(-1, in_dim).contiguous()
        M = x.shape[0]
        out = torch.empty((M, out_dim), device=x.device, dtype=x.dtype)

        small = M * out_dim * in_dim <= 64 * 64 * 64
        has_bias = self.bias is not None
        act = bool(self.activation)
        if small:
            BLOCK_M = triton.next_power_of_2(M)
            BLOCK_N = triton.next_power_of_2(out_dim)
            BLOCK_K = triton.next_power_of_2(in_dim)
            _linear_kernel[(1,)](
                x, self.weight, self.bias if has_bias else x, out,
                M, out_dim, in_dim,
                self.scale, self.lr_mul, 0.2, 2 ** 0.5,
                HAS_BIAS=has_bias, ACT=act,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=1, num_stages=1,
            )
        else:
            BLOCK_M = 64
            BLOCK_N = max(16, triton.next_power_of_2(out_dim))
            BLOCK_K = max(16, triton.next_power_of_2(in_dim))
            _dot_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(out_dim, BLOCK_N))](
                x, self.weight, self.bias if has_bias else x, out,
                M, out_dim, in_dim,
                self.scale, self.lr_mul, 0.2, 2 ** 0.5,
                HAS_BIAS=has_bias, ACT=act,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=2,
            )
        return out.reshape(*input.shape[:-1], out_dim)


@triton.jit
def _dot_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    scale, lr_mul, neg_slope, act_scale,
    HAS_BIAS: tl.constexpr, ACT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        kk = k + offs_k
        x = tl.load(x_ptr + offs_m[:, None] * K + kk[None, :],
                    mask=(offs_m[:, None] < M) & (kk[None, :] < K), other=0.0)
        w = tl.load(w_ptr + offs_n[None, :] * K + kk[:, None],
                    mask=(offs_n[None, :] < N) & (kk[:, None] < K), other=0.0)
        acc += tl.dot(x, w)
    acc = acc * scale
    if HAS_BIAS:
        b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += (b * lr_mul)[None, :]
    if ACT:
        acc = tl.where(acc >= 0, acc, acc * neg_slope) * act_scale
    out_off = offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptr + out_off, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
