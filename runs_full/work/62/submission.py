import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ls_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_wn, stride_om,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N
    k_mask = offs_k < K

    x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                mask=m_mask[:, None] & k_mask[None, :], other=0.0)
    w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                mask=n_mask[:, None] & k_mask[None, :], other=0.0)
    acc = tl.sum(x[:, None, :] * w[None, :, :], axis=2)

    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias[None, :]

    out = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(acc)))
    out_off = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_off, out, mask=m_mask[:, None] & n_mask[None, :])


class Linear_soft_plusNew(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=bias)
        self.activation = nn.Softplus()

    def forward(self, x):
        orig_shape = x.shape
        K = orig_shape[-1]
        x2d = x.reshape(-1, K).contiguous()
        M = x2d.shape[0]
        N = self.linear.weight.shape[0]
        w = self.linear.weight.contiguous()
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        has_bias = self.linear.bias is not None
        b = self.linear.bias if has_bias else x2d

        BLOCK_N = triton.next_power_of_2(N)
        BLOCK_K = triton.next_power_of_2(K)
        BLOCK_M = min(triton.next_power_of_2(M), 256)
        grid = (triton.cdiv(M, BLOCK_M),)
        _ls_kernel[grid](
            x2d, w, b, out,
            M, N, K,
            x2d.stride(0), w.stride(0), out.stride(0),
            HAS_BIAS=has_bias,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=1, num_stages=2,
        )
        return out.reshape(*orig_shape[:-1], N)
