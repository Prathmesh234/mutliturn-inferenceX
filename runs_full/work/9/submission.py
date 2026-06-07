import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mlp_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
                M, K, P,
                BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_P: tl.constexpr):
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    rp = tl.arange(0, BLOCK_P)
    m_mask = rm < M
    k_mask = rk < K
    p_mask = rp < P

    x = tl.load(x_ptr + rm[:, None] * K + rk[None, :],
                mask=m_mask[:, None] & k_mask[None, :], other=0.0)
    w1 = tl.load(w1_ptr + rk[:, None] * P + rp[None, :],
                 mask=k_mask[:, None] & p_mask[None, :], other=0.0)
    acc1 = tl.dot(x, w1, out_dtype=tl.float32)
    b1 = tl.load(b1_ptr + rp, mask=p_mask, other=0.0)
    acc1 += b1[None, :]
    h = 0.5 * acc1 * (1.0 + tl.erf(acc1 * 0.7071067811865476))
    h = h.to(x.dtype)
    w2 = tl.load(w2_ptr + rp[:, None] * K + rk[None, :],
                 mask=p_mask[:, None] & k_mask[None, :], other=0.0)
    acc2 = tl.dot(h, w2, out_dtype=tl.float32)
    b2 = tl.load(b2_ptr + rk, mask=k_mask, other=0.0)
    acc2 += b2[None, :]
    tl.store(out_ptr + rm[:, None] * K + rk[None, :], acc2,
             mask=m_mask[:, None] & k_mask[None, :])


class Conv1d(nn.Module):
    def __init__(self, nf, nx, stdev=0.02):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.stdev = stdev
        self.w = nn.Parameter(torch.normal(size=[1, self.nx, self.nf], mean=0.0, std=self.stdev))
        self.b = nn.Parameter(torch.zeros([self.nf]))


class MlpNew(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.conv_fc = Conv1d(self.proj_dim, self.input_dim)
        self.conv_proj = Conv1d(self.input_dim, self.proj_dim)

    def forward(self, x):
        shape = x.size()
        K = self.input_dim
        P = self.proj_dim
        xf = x.reshape(-1, K).contiguous()
        M = xf.shape[0]
        out = torch.empty_like(xf)
        w1 = self.conv_fc.w.reshape(K, P).contiguous()
        w2 = self.conv_proj.w.reshape(P, K).contiguous()
        BLOCK_M = triton.next_power_of_2(M)
        BLOCK_K = max(16, triton.next_power_of_2(K))
        BLOCK_P = max(16, triton.next_power_of_2(P))
        grid = (triton.cdiv(M, BLOCK_M),)
        _mlp_kernel[grid](xf, w1, self.conv_fc.b, w2, self.conv_proj.b, out,
                          M, K, P, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_P=BLOCK_P,
                          num_warps=2, num_stages=1)
        return out.reshape(shape[:-1] + (K,))
