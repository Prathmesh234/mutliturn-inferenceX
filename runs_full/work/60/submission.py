import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _attn_kernel(M_ptr, W_ptr, pool_ptr, alpha_ptr,
                 seq, batch, vec,
                 stride_s, stride_b, stride_v,
                 BLOCK_S: tl.constexpr, BLOCK_V: tl.constexpr):
    b = tl.program_id(0)
    offs_s = tl.arange(0, BLOCK_S)
    offs_v = tl.arange(0, BLOCK_V)
    mask_s = offs_s < seq
    mask_v = offs_v < vec
    m_ptrs = M_ptr + offs_s[:, None] * stride_s + b * stride_b + offs_v[None, :] * stride_v
    m = tl.load(m_ptrs, mask=mask_s[:, None] & mask_v[None, :], other=0.0)
    w = tl.load(W_ptr + offs_v, mask=mask_v, other=0.0)
    scale = tl.sum(m * w[None, :], axis=1)
    scale = tl.where(mask_s, scale, -float('inf'))
    mx = tl.max(scale, axis=0)
    e = tl.exp(scale - mx)
    e = tl.where(mask_s, e, 0.0)
    ssum = tl.sum(e, axis=0)
    alpha = e / ssum
    pool = tl.sum(alpha[:, None] * m, axis=0)
    tl.store(pool_ptr + b * vec + offs_v, pool, mask=mask_v)
    tl.store(alpha_ptr + b * seq + offs_s, alpha, mask=mask_s)


class SimpleAttentionNew(nn.Module):
    def __init__(self, input_dim):
        super(SimpleAttentionNew, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim, 1, bias=False)

    def forward(self, M, x=None):
        seq, batch, vec = M.shape
        M = M.contiguous()
        pool = torch.empty((batch, vec), device=M.device, dtype=M.dtype)
        alpha = torch.empty((batch, 1, seq), device=M.device, dtype=M.dtype)
        w = self.scalar.weight.view(-1).contiguous()
        BLOCK_S = triton.next_power_of_2(seq)
        BLOCK_V = triton.next_power_of_2(vec)
        _attn_kernel[(batch,)](M, w, pool, alpha,
                               seq, batch, vec,
                               M.stride(0), M.stride(1), M.stride(2),
                               BLOCK_S=BLOCK_S, BLOCK_V=BLOCK_V, num_warps=1)
        return pool, alpha
