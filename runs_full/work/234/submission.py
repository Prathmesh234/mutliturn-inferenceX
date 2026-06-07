import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                   M, N, K,
                   stride_xm, stride_xk,
                   stride_wn, stride_wk,
                   stride_om, stride_on,
                   APPLY_RELU: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        kk = k + offs_k
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk,
                    mask=(offs_m[:, None] < M) & (kk[None, :] < K), other=0.0)
        wt = tl.load(w_ptr + kk[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                     mask=(kk[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(x, wt, allow_tf32=False)
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b[None, :]
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _linear(x, w, b, relu, BLOCK_N, BLOCK_K, num_warps):
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_M = triton.next_power_of_2(M)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _linear_kernel[grid](x, w, b, out, M, N, K,
                         x.stride(0), x.stride(1),
                         w.stride(0), w.stride(1),
                         out.stride(0), out.stride(1),
                         APPLY_RELU=relu,
                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                         num_warps=num_warps, num_stages=2)
    return out


class CriticNew(nn.Module):
    def __init__(self, input_size):
        super(CriticNew, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        orig_shape = x.shape
        K = orig_shape[-1]
        x2 = x.reshape(-1, K).contiguous()
        h1 = _linear(x2, self.fc1.weight, self.fc1.bias, True, 128, 16, 4)
        h2 = _linear(h1, self.fc2.weight, self.fc2.bias, True, 128, 64, 4)
        out = _linear(h2, self.fc3.weight, self.fc3.bias, False, 16, 64, 4)
        return out.reshape(*orig_shape[:-1], 1)
