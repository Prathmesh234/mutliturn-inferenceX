import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linemb_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                   M, N, K, scale,
                   stride_xm, stride_xk,
                   stride_wn, stride_wk,
                   stride_om, stride_on,
                   BLOCK_M: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    mask_m = offs_m < M
    x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=mask_m[:, None] & (offs_k[None, :] < K), other=0.0)
    w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
    acc = tl.sum(x[:, None, :] * w[None, :, :], axis=2)
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = (acc + b[None, :]) * scale
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc, mask=mask_m[:, None] & (offs_n[None, :] < N))


class LinearEmbeddingNew(nn.Module):
    def __init__(self, inp_size, d_model):
        super(LinearEmbeddingNew, self).__init__()
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        scale = math.sqrt(self.d_model)
        orig_shape = x.shape
        K = orig_shape[-1]
        M = x.numel() // K
        N = self.d_model
        x2 = x.reshape(M, K)
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        w = self.lut.weight
        b = self.lut.bias
        BLOCK_M = 256
        BN = triton.next_power_of_2(N)
        BK = triton.next_power_of_2(K)
        grid = (triton.cdiv(M, BLOCK_M),)
        _linemb_kernel[grid](x2, w, b, out, M, N, K, scale,
                             x2.stride(0), x2.stride(1),
                             w.stride(0), w.stride(1),
                             out.stride(0), out.stride(1),
                             BLOCK_M=BLOCK_M, BN=BN, BK=BK,
                             num_warps=8)
        return out.reshape(*orig_shape[:-1], N)
