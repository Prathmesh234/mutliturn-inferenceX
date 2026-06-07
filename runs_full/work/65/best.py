import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_lrelu_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                         M, N, K, slope,
                         stride_xm, stride_xk,
                         stride_wn, stride_wk,
                         stride_om, stride_on,
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
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + kk[None, :] * stride_wk,
                    mask=(offs_n[:, None] < N) & (kk[None, :] < K), other=0.0)
        acc += tl.dot(x, tl.trans(w))

    if b_ptr is not None:
        b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += b[None, :]

    acc = tl.where(acc >= 0, acc, acc * slope)

    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class Linear_leaky_reluNew(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=bias)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        slope = self.activation.negative_slope
        w = self.linear.weight
        b = self.linear.bias
        N, K = w.shape
        x2 = x.reshape(-1, K)
        M = x2.shape[0]
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        BLOCK_M = 64
        BLOCK_N = max(16, triton.next_power_of_2(N))
        BLOCK_K = max(16, triton.next_power_of_2(K))
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _linear_lrelu_kernel[grid](
            x2, w, b if b is not None else x2, out,
            M, N, K, slope,
            x2.stride(0), x2.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=1, num_stages=2)
        return out.reshape(*x.shape[:-1], N)
