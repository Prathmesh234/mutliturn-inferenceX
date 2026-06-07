import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_tanh_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                        M, N, K,
                        stride_xm, stride_xk,
                        stride_wn, stride_wk,
                        stride_om, stride_on,
                        HAS_BIAS: tl.constexpr,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
    w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)

    prod = x[:, None, :] * w[None, :, :]
    acc = tl.sum(prod, axis=2)

    if HAS_BIAS:
        b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += b[None, :]

    e = tl.exp(2 * acc)
    acc = (e - 1) / (e + 1)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc, mask=out_mask)


class Linear_tanhNew(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=bias)
        self.activation = nn.Tanh()

    def forward(self, x):
        orig_shape = x.shape
        K = orig_shape[-1]
        x2d = x.reshape(-1, K).contiguous()
        M = x2d.shape[0]
        N = self.linear.weight.shape[0]
        w = self.linear.weight
        b = self.linear.bias
        has_bias = b is not None
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        BLOCK_M = triton.next_power_of_2(M)
        BLOCK_N = triton.next_power_of_2(N)
        BLOCK_K = triton.next_power_of_2(K)
        grid = (triton.cdiv(M, BLOCK_M),)
        _linear_tanh_kernel[grid](
            x2d, w, b if has_bias else x2d, out,
            M, N, K,
            x2d.stride(0), x2d.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            HAS_BIAS=has_bias,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=2, num_stages=1,
        )
        return out.reshape(*orig_shape[:-1], N)


