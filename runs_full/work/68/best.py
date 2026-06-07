import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_sigmoid_kernel(x_ptr, w_ptr, b_ptr, out_ptr, M, K: tl.constexpr,
                           N: tl.constexpr, HAS_BIAS: tl.constexpr,
                           BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_n = tl.arange(0, N)

    acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    for k in range(K):
        a = tl.load(x_ptr + offs_m * K + k, mask=mask_m, other=0.0).to(tl.float32)
        wk = tl.load(w_ptr + offs_n * K + k).to(tl.float32)
        acc += a[:, None] * wk[None, :]
    if HAS_BIAS:
        b = tl.load(b_ptr + offs_n).to(tl.float32)
        acc += b[None, :]
    acc = 1.0 / (1.0 + tl.exp(-acc))
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_m[:, None])


class Linear_sigmoidNew(nn.Module):

    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=bias)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        dim_in = self.linear.in_features
        dim_out = self.linear.out_features
        x_flat = x.reshape(-1, dim_in)
        M = x_flat.shape[0]
        out = torch.empty((M, dim_out), device=x.device, dtype=torch.float32)
        has_bias = self.linear.bias is not None
        b = self.linear.bias if has_bias else x_flat
        BLOCK_M = triton.next_power_of_2(M)
        grid = (1,)
        _linear_sigmoid_kernel[grid](x_flat, self.linear.weight, b, out, M,
                                     dim_in, dim_out, has_bias,
                                     BLOCK_M=BLOCK_M, num_warps=1)
        return out.reshape(*x.shape[:-1], dim_out)
