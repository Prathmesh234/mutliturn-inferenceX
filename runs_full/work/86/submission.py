import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused12_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, c_ptr,
                    M, N, KIN, H,
                    stride_xm, stride_xk,
                    stride_w1h, stride_w1k,
                    stride_w2n, stride_w2h,
                    stride_cm, stride_cn,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_H: tl.constexpr, BLOCK_KIN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kin = tl.arange(0, BLOCK_KIN)
    offs_h = tl.arange(0, BLOCK_H)

    # load x[BLOCK_M, BLOCK_KIN] once
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_kin[None, :] * stride_xk
    x_mask = (offs_m[:, None] < M) & (offs_kin[None, :] < KIN)
    x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for h in range(0, H, BLOCK_H):
        cur_h = h + offs_h
        h_mask = cur_h < H
        # a1 tile = relu(x @ W1[cur_h]^T + b1[cur_h]) -> [BLOCK_M, BLOCK_H]
        w1_ptrs = w1_ptr + cur_h[None, :] * stride_w1h + offs_kin[:, None] * stride_w1k
        w1_mask = (cur_h[None, :] < H) & (offs_kin[:, None] < KIN)
        w1_tile = tl.load(w1_ptrs, mask=w1_mask, other=0.0)
        a1 = tl.dot(x_tile, w1_tile)
        b1 = tl.load(b1_ptr + cur_h, mask=h_mask, other=0.0)
        a1 = tl.maximum(a1 + b1[None, :], 0.0)
        a1 = a1.to(x_tile.dtype)
        # acc += a1 @ W2[offs_n, cur_h]^T
        w2_ptrs = w2_ptr + offs_n[None, :] * stride_w2n + cur_h[:, None] * stride_w2h
        w2_mask = (offs_n[None, :] < N) & (cur_h[:, None] < H)
        w2_tile = tl.load(w2_ptrs, mask=w2_mask, other=0.0)
        acc += tl.dot(a1, w2_tile)

    b2 = tl.load(b2_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = tl.maximum(acc + b2[None, :], 0.0)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(x_tile.dtype), mask=c_mask)


@triton.jit
def _linear_kernel(a_ptr, w_ptr, b_ptr, c_ptr,
                   M, N, K,
                   stride_am, stride_ak,
                   stride_wn, stride_wk,
                   stride_cm, stride_cn,
                   APPLY_RELU: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_rem = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_rem)
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < k_rem)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(a, w)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _linear(a, w, b, relu):
    M, K = a.shape
    N = w.shape[0]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M = max(16, triton.next_power_of_2(M)) if M < 64 else 64
    BLOCK_N = max(16, triton.next_power_of_2(N)) if N < 128 else 128
    BLOCK_K = 64 if K >= 64 else 16
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _linear_kernel[grid](a, w, b, c, M, N, K,
                         a.stride(0), a.stride(1),
                         w.stride(0), w.stride(1),
                         c.stride(0), c.stride(1),
                         APPLY_RELU=relu,
                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                         num_warps=4, num_stages=2)
    return c


class LargeNNNew(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.l1 = nn.Linear(in_channels, 1024)
        self.l2 = nn.Linear(1024, 1024)
        self.l3 = nn.Linear(1024, out_channels)

    def forward(self, xb):
        orig_shape = xb.shape
        KIN = orig_shape[-1]
        x = xb.reshape(-1, KIN).contiguous()
        M = x.shape[0]
        H = self.l1.weight.shape[0]   # 1024
        N2 = self.l2.weight.shape[0]  # 1024

        a2 = torch.empty((M, N2), device=x.device, dtype=x.dtype)
        BLOCK_M = max(16, triton.next_power_of_2(M)) if M < 64 else 64
        BLOCK_N = 64
        BLOCK_H = 64
        BLOCK_KIN = max(16, triton.next_power_of_2(KIN))
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N2, BLOCK_N))
        _fused12_kernel[grid](
            x, self.l1.weight, self.l1.bias,
            self.l2.weight, self.l2.bias, a2,
            M, N2, KIN, H,
            x.stride(0), x.stride(1),
            self.l1.weight.stride(0), self.l1.weight.stride(1),
            self.l2.weight.stride(0), self.l2.weight.stride(1),
            a2.stride(0), a2.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H,
            BLOCK_KIN=BLOCK_KIN, num_warps=4, num_stages=3)

        out = _linear(a2, self.l3.weight, self.l3.bias, False)
        return out.reshape(*orig_shape[:-1], out.shape[-1])
