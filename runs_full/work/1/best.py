import torch
import torch.nn as nn
import math
import triton
import triton.language as tl


@triton.jit
def _linear_embed_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                         M, scale,
                         K: tl.constexpr, N: tl.constexpr,
                         BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_k = tl.arange(0, K)
    # load x block [BLOCK_M, K]
    x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=mask_m[:, None], other=0.0)
    for n in tl.static_range(N):
        w = tl.load(w_ptr + n * K + offs_k)            # [K]
        b = tl.load(b_ptr + n)
        acc = tl.sum(x * w[None, :], axis=1) + b
        acc = acc * scale
        tl.store(out_ptr + offs_m * N + n, acc, mask=mask_m)


class LinearEmbeddingNew(nn.Module):

    def __init__(self, inp_size, d_model):
        super(LinearEmbeddingNew, self).__init__()
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        scale = math.sqrt(self.d_model)
        K = self.lut.in_features
        N = self.lut.out_features
        x_flat = x.reshape(-1, K).contiguous()
        M = x_flat.shape[0]
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        BLOCK_M = 256
        grid = (triton.cdiv(M, BLOCK_M),)
        _linear_embed_kernel[grid](
            x_flat, self.lut.weight, self.lut.bias, out,
            M, scale, K=K, N=N, BLOCK_M=BLOCK_M, num_warps=4)
        return out.reshape(*x.shape[:-1], N)
