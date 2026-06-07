import math
import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _conv2d_kernel(
    inp_ptr, w_ptr, out_ptr, bias_ptr,
    N, IC, OC, IH, IW, OH, OW, KH, KW,
    stride, padding, scale,
    M, K,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    n_idx = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        ic = offs_k // (KH * KW)
        rem2 = offs_k % (KH * KW)
        kh = rem2 // KW
        kw = rem2 % KW

        ih = oh[:, None] * stride - padding + kh[None, :]
        iw = ow[:, None] * stride - padding + kw[None, :]

        a_idx = ((n_idx[:, None] * IC + ic[None, :]) * IH + ih) * IW + iw
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K) & \
                 (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
        a = tl.load(inp_ptr + a_idx, mask=a_mask, other=0.0)

        b_idx = offs_k[:, None] * 1 + offs_n[None, :] * K
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < OC)
        b = tl.load(w_ptr + b_idx, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    acc = acc * scale
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < OC, other=0.0)
        acc += bias[None, :]

    out_idx = ((n_idx[:, None] * OC + offs_n[None, :]) * OH + oh[:, None]) * OW + ow[:, None]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < OC)
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


class EqualConv2dNew(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
        padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel,
            kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        input = input.contiguous()
        N, IC, IH, IW = input.shape
        OC, _, KH, KW = self.weight.shape
        OH = (IH + 2 * self.padding - KH) // self.stride + 1
        OW = (IW + 2 * self.padding - KW) // self.stride + 1
        out = torch.empty((N, OC, OH, OW), device=input.device, dtype=input.dtype)
        M = N * OH * OW
        K = IC * KH * KW
        BM = max(16, triton.next_power_of_2(M))
        BN = max(16, triton.next_power_of_2(OC))
        BK = max(16, triton.next_power_of_2(K))
        grid = (triton.cdiv(M, BM), triton.cdiv(OC, BN))
        w = self.weight.contiguous()
        bias = self.bias if self.bias is not None else input
        _conv2d_kernel[grid](
            input, w, out, bias,
            N, IC, OC, IH, IW, OH, OW, KH, KW,
            self.stride, self.padding, self.scale,
            M, K,
            HAS_BIAS=self.bias is not None,
            BM=BM, BN=BN, BK=BK, num_warps=2, num_stages=2,
        )
        return out
