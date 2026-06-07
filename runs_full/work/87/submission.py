import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _tanh(x):
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def _lrn_kernel(x_ptr, w_ptr, b_ptr, h_ptr, out_ptr,
                B, I, H,
                stride_xb, stride_xi,
                stride_wi, stride_wn,
                stride_hb, stride_hh,
                stride_ob, stride_oh,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_K: tl.constexpr, N3: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n3 = tl.arange(0, N3)

    acc = tl.zeros((BLOCK_M, N3), dtype=tl.float32)
    for k in range(0, I, BLOCK_K):
        kk = k + offs_k
        a = tl.load(x_ptr + offs_m[:, None] * stride_xb + kk[None, :] * stride_xi,
                    mask=(offs_m[:, None] < B) & (kk[None, :] < I), other=0.0)
        w = tl.load(w_ptr + kk[:, None] * stride_wi + offs_n3[None, :] * stride_wn,
                    mask=(kk[:, None] < I) & (offs_n3[None, :] < 3 * H), other=0.0)
        acc += tl.dot(a, w)

    b = tl.load(b_ptr + offs_n3, mask=offs_n3 < 3 * H, other=0.0)
    acc += b[None, :]

    offs_n = tl.arange(0, BLOCK_N)
    nmask = (offs_m[:, None] < B) & (offs_n[None, :] < H)
    h_ = tl.load(h_ptr + offs_m[:, None] * stride_hb + offs_n[None, :] * stride_hh,
                 mask=nmask, other=0.0)

    accp = tl.sum(tl.where(offs_n3[None, :] == 0, 0.0, 0.0), axis=1)  # placeholder removed
    p = tl.sum(tl.where(offs_n3[None, None, :] == 0, acc[:, :, None], 0.0), axis=2)
    tl.store(out_ptr, p)
