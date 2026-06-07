import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import triton
import triton.language as tl


@triton.jit
def _nac_kernel(x_ptr, w_hat_ptr, m_hat_ptr, out_ptr,
                M, K, N,
                BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    k_mask = offs_k < K
    n_mask = offs_n < N

    x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=m_mask[:, None] & k_mask[None, :], other=0.0)

    wptr = offs_n[None, :] * K + offs_k[:, None]
    wmask = k_mask[:, None] & n_mask[None, :]
    w_hat = tl.load(w_hat_ptr + wptr, mask=wmask, other=0.0)
    m_hat = tl.load(m_hat_ptr + wptr, mask=wmask, other=0.0)
    tanh_w = 2.0 * tl.sigmoid(2.0 * w_hat) - 1.0
    w_t = tanh_w * tl.sigmoid(m_hat)

    acc = tl.dot(x, w_t, allow_tf32=False)

    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


class NACNew(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.W_hat = Parameter(torch.Tensor(n_out, n_in))
        self.M_hat = Parameter(torch.Tensor(n_out, n_in))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_hat)
        nn.init.kaiming_uniform_(self.M_hat)

    def forward(self, input):
        n_out, n_in = self.W_hat.shape
        x = input.contiguous()
        M = x.numel() // n_in
        x2 = x.view(M, n_in)
        out = torch.empty((M, n_out), device=x.device, dtype=x.dtype)

        BLOCK_M = 64
        BLOCK_K = max(16, triton.next_power_of_2(n_in))
        BLOCK_N = max(16, triton.next_power_of_2(n_out))
        grid = (triton.cdiv(M, BLOCK_M),)
        _nac_kernel[grid](x2, self.W_hat, self.M_hat, out,
                          M, n_in, n_out,
                          BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
                          num_warps=2)
        return out.view(*input.shape[:-1], n_out)
