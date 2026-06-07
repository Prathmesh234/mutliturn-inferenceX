import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _pool_linear_kernel(feat_ptr, w_ptr, b_ptr, out_ptr,
                        L, R, K, N, rows_per_i,
                        BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    m = tl.program_id(0)
    i = m // rows_per_i
    r_base = (m % rows_per_i) * K
    base = i * L * R + r_base

    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_K,), tl.float32)
    for l in range(0, L):
        x = tl.load(feat_ptr + base + l * R + offs_k, mask=mask_k, other=0.0)
        acc += x
    x = acc / L

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :], other=0.0)
    out = tl.sum(w * x[None, :], axis=1)
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    out = out + b
    tl.store(out_ptr + m * N + offs_n, out, mask=mask_n)


class ModelNew(nn.Module):
    def __init__(self, input_dim, output_class_num, **kwargs):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_class_num)

    def forward(self, features):
        L = features.shape[1]
        K = features.shape[-1]
        d0 = features.shape[0]
        feat = features.contiguous()
        R = feat.numel() // (d0 * L)
        rows_per_i = R // K
        M = d0 * rows_per_i
        N = self.linear.out_features

        out_shape = list(features.shape[:1]) + list(features.shape[2:-1]) + [N]
        out = torch.empty(M, N, device=features.device, dtype=torch.float32)

        BLOCK_K = triton.next_power_of_2(K)
        BLOCK_N = triton.next_power_of_2(N)
        grid = (M,)
        _pool_linear_kernel[grid](feat, self.linear.weight, self.linear.bias, out,
                                  L, R, K, N, rows_per_i,
                                  BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N, num_warps=1)
        return out.reshape(out_shape).to(features.dtype)
