import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _maxout_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                   M, K,
                   stride_xm, stride_xk,
                   stride_wk, stride_wn,
                   stride_om, stride_oo,
                   out_feature,
                   BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
                   BLOCK_O: tl.constexpr, POOL: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    BLOCK_N: tl.constexpr = BLOCK_O * POOL
    offs_n = tl.arange(0, BLOCK_N)
    o_idx = offs_n // POOL
    n_mask = o_idx < out_feature

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, BLOCK_K):
        k_valid = k < K
        x_k = tl.load(x_ptr + offs_m * stride_xm + k * stride_xk,
                      mask=m_mask & k_valid, other=0.0)
        wt_k = tl.load(w_ptr + k * stride_wk + offs_n * stride_wn,
                       mask=n_mask & k_valid, other=0.0)
        acc += x_k[:, None] * wt_k[None, :]

    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc += bias[None, :]
    acc = tl.where(n_mask[None, :], acc, -float('inf'))

    acc3 = tl.reshape(acc, (BLOCK_M, BLOCK_O, POOL))
    res = tl.max(acc3, axis=2)

    offs_o = tl.arange(0, BLOCK_O)
    o_mask = offs_o < out_feature
    store_ptr = out_ptr + offs_m[:, None] * stride_om + offs_o[None, :] * stride_oo
    tl.store(store_ptr, res, mask=m_mask[:, None] & o_mask[None, :])


class maxoutNew(nn.Module):
    def __init__(self, in_feature, out_feature, pool_size):
        super(maxoutNew, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.pool_size = pool_size
        self.linear = nn.Linear(in_feature, out_feature * pool_size)

    def forward(self, x):
        K = self.in_feature
        x2 = x.reshape(-1, K)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        M = x2.shape[0]
        w = self.linear.weight
        b = self.linear.bias
        out = torch.empty((M, self.out_feature), device=x.device, dtype=x.dtype)

        BLOCK_M = 128
        BLOCK_K = triton.next_power_of_2(K)
        BLOCK_O = triton.next_power_of_2(self.out_feature)
        grid = (triton.cdiv(M, BLOCK_M),)
        _maxout_kernel[grid](
            x2, w, b, out,
            M, K,
            x2.stride(0), x2.stride(1),
            w.stride(1), w.stride(0),
            out.stride(0), out.stride(1),
            self.out_feature,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            BLOCK_O=BLOCK_O, POOL=self.pool_size,
            num_warps=1,
        )
        return out
