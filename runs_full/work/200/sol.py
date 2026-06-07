import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
import triton
import triton.language as tl


@triton.jit
def _nalu_kernel(X_ptr, Wh_ptr, Mh_ptr, G_ptr, Out_ptr,
                 M, N_IN: tl.constexpr, N_OUT: tl.constexpr,
                 eps,
                 BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_k = offs_k < N_IN
    mask_n = offs_n < N_OUT

    # Load X (BLOCK_M, BLOCK_K); pad with 1.0 so log() is finite (weights pad=0)
    x_ptrs = X_ptr + offs_m[:, None] * N_IN + offs_k[None, :]
    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=1.0)

    # Load and compute weights (BLOCK_N, BLOCK_K) = tanh(W_hat)*sigmoid(M_hat)
    w_ptrs = offs_n[:, None] * N_IN + offs_k[None, :]
    wmask = mask_n[:, None] & mask_k[None, :]
    wh = tl.load(Wh_ptr + w_ptrs, mask=wmask, other=0.0)
    mh = tl.load(Mh_ptr + w_ptrs, mask=wmask, other=0.0)
    tanh_wh = 2.0 * tl.sigmoid(2.0 * wh) - 1.0
    weights = tanh_wh * tl.sigmoid(mh)  # pad rows/cols -> 0

    # Gate weights G (BLOCK_K,)
    gw = tl.load(G_ptr + offs_k, mask=mask_k, other=0.0)

    x3 = x[:, None, :]
    w3 = weights[None, :, :]

    # additive path: a = X @ W.T
    a = tl.sum(x3 * w3, axis=2)  # (BLOCK_M, BLOCK_N)

    # multiplicative path
    logx = tl.log(tl.abs(x) + eps)
    m_pre = tl.sum(logx[:, None, :] * w3, axis=2)
    m_val = tl.exp(m_pre)

    # gate
    g_pre = tl.sum(x * gw[None, :], axis=1)  # (BLOCK_M,)
    g = tl.sigmoid(g_pre)[:, None]

    out = g * a + (1.0 - g) * m_val

    out_ptrs = Out_ptr + offs_m[:, None] * N_OUT + offs_n[None, :]
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


class NAC(Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.W_hat = Parameter(torch.Tensor(n_out, n_in))
        self.M_hat = Parameter(torch.Tensor(n_out, n_in))


class NALUNew(Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.NAC = NAC(n_in, n_out)
        self.G = Parameter(torch.Tensor(1, n_in))
        self.eps = 1e-06
        self.n_in = n_in
        self.n_out = n_out

    def forward(self, input):
        x = input.contiguous()
        orig_shape = x.shape
        M = x.numel() // self.n_in
        x2 = x.view(M, self.n_in)
        out = torch.empty((M, self.n_out), device=x.device, dtype=x.dtype)

        BLOCK_K = triton.next_power_of_2(self.n_in)
        BLOCK_N = triton.next_power_of_2(self.n_out)
        BLOCK_M = 64
        grid = (triton.cdiv(M, BLOCK_M),)
        _nalu_kernel[grid](
            x2, self.NAC.W_hat, self.NAC.M_hat, self.G, out,
            M, self.n_in, self.n_out, self.eps,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=1,
        )
        return out.view(*orig_shape[:-1], self.n_out)
