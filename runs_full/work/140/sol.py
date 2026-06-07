import math
import torch
import torch.nn as nn
from torch.nn import init
import triton
import triton.language as tl


@triton.jit
def _bayes_kernel(x_ptr, wmu_ptr, wls_ptr, bias_ptr, eps_ptr, out_ptr,
                  M, N, K,
                  HAS_BIAS: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    mask_m = rm < M
    mask_n = rn < N
    mask_k = rk < K

    # x tile [BLOCK_M, BLOCK_K]
    x = tl.load(x_ptr + rm[:, None] * K + rk[None, :],
                mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    x = x.to(tl.float32)
    x2 = x * x

    # weight tiles [BLOCK_K, BLOCK_N]
    w_off = rk[:, None] + rn[None, :] * K
    w_mask = mask_k[:, None] & mask_n[None, :]
    wmu = tl.load(wmu_ptr + w_off, mask=w_mask, other=0.0).to(tl.float32)
    wls = tl.load(wls_ptr + w_off, mask=w_mask, other=0.0).to(tl.float32)
    ws = tl.exp(wls)
    ws2 = ws * ws

    acc_mu = tl.dot(x, wmu, out_dtype=tl.float32)
    acc_sig = tl.dot(x2, ws2, out_dtype=tl.float32)

    if HAS_BIAS:
        b = tl.load(bias_ptr + rn, mask=mask_n, other=0.0).to(tl.float32)
        acc_mu = acc_mu + b[None, :]

    sigma = tl.sqrt(acc_sig + 1e-08)
    eps = tl.load(eps_ptr + rm[:, None] * N + rn[None, :],
                  mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
    out = acc_mu + sigma * eps

    tl.store(out_ptr + rm[:, None] * N + rn[None, :], out,
             mask=mask_m[:, None] & mask_n[None, :])


class BayesLinearNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 log_sigma_prior=-5, mu_prior=-1):
        super(BayesLinearNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.w_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.mu_prior_init = mu_prior
        self.log_sigma_prior_init = log_sigma_prior
        if bias is True:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        init.uniform_(self.w_log_sigma, self.log_sigma_prior_init - 0.1,
                      self.log_sigma_prior_init)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        orig_shape = input.shape
        K = self.in_features
        N = self.out_features
        x = input.contiguous().view(-1, K)
        M = x.shape[0]

        out_shape = orig_shape[:-1] + (N,)
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        eps = torch.randn(out_shape, device=x.device, dtype=x.dtype).view(-1, N)

        BLOCK_M = 64
        BLOCK_N = max(16, triton.next_power_of_2(N))
        BLOCK_K = max(16, triton.next_power_of_2(K))
        grid = (triton.cdiv(M, BLOCK_M),)

        _bayes_kernel[grid](
            x, self.w_mu, self.w_log_sigma,
            self.bias if self.bias is not None else x,
            eps, out, M, N, K,
            HAS_BIAS=self.bias is not None,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )
        return out.view(out_shape)
