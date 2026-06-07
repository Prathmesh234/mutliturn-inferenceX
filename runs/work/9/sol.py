import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _mlp_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
                M, K, N,
                BLOCK_M: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BK)
    offs_n = tl.arange(0, BN)

    m_mask = offs_m < M
    k_mask = offs_k < K
    n_mask = offs_n < N

    # load x [BLOCK_M, BK]
    x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=m_mask[:, None] & k_mask[None, :], other=0.0)
    # W1 [BK, BN]
    w1 = tl.load(w1_ptr + offs_k[:, None] * N + offs_n[None, :],
                 mask=k_mask[:, None] & n_mask[None, :], other=0.0)
    h = tl.dot(x, w1, out_dtype=tl.float32)
    b1 = tl.load(b1_ptr + offs_n, mask=n_mask, other=0.0)
    h = h + b1[None, :]
    # exact gelu
    h = h * 0.5 * (1.0 + libdevice.erf(h * 0.7071067811865476))
    h = h.to(x.dtype)

    # W2 [BN, BK]
    w2 = tl.load(w2_ptr + offs_n[:, None] * K + offs_k[None, :],
                 mask=n_mask[:, None] & k_mask[None, :], other=0.0)
    out = tl.dot(h, w2, out_dtype=tl.float32)
    b2 = tl.load(b2_ptr + offs_k, mask=k_mask, other=0.0)
    out = out + b2[None, :]

    tl.store(out_ptr + offs_m[:, None] * K + offs_k[None, :], out,
             mask=m_mask[:, None] & k_mask[None, :])


class Conv1d(nn.Module):
    def __init__(self, nf, nx, stdev=0.02):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.stdev = stdev
        self.w = nn.Parameter(torch.normal(size=[1, self.nx, self.nf], mean=0.0, std=self.stdev))
        self.b = nn.Parameter(torch.zeros([self.nf]))

    def forward(self, x):
        shape = x.size()
        start, nx = shape[:-1], shape[-1]
        return torch.reshape(torch.matmul(torch.reshape(x, [-1, nx]),
                             torch.reshape(self.w, [-1, self.nf])) + self.b, start + (self.nf,))


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
        N = self.proj_dim
        xf = x.reshape(-1, K)
        M = xf.shape[0]
        w1 = self.conv_fc.w.reshape(K, N)
        w2 = self.conv_proj.w.reshape(N, K)
        out = torch.empty_like(xf)
        BLOCK_M = 64
        BK = max(16, triton.next_power_of_2(K))
        BN = max(16, triton.next_power_of_2(N))
        grid = (triton.cdiv(M, BLOCK_M),)
        _mlp_kernel[grid](xf, w1, self.conv_fc.b, w2, self.conv_proj.b, out,
                          M, K, N, BLOCK_M=BLOCK_M, BK=BK, BN=BN,
                          num_warps=2, num_stages=1)
        return out.reshape(shape)
