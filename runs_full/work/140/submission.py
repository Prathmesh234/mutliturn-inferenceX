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

    x = tl.load(x_ptr + rm[:, None] * K + rk[None, :],
                mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    x = x.to(tl.float32)
    x2 = x * x

    w_off = rk[:, None] + rn[None, :] * K
    w_mask = mask_k[:, None] & mask_n[None, :]
    wmu = tl.load(wmu_ptr + w_off, mask=w_mask, other=0.0).to(tl.float32)
    wls = tl.load(wls_ptr + w_off, mask=w_mask, other=0.0).to(tl.float32)
    ws = tl.exp(wls)
    ws2 = ws * ws

    acc_mu = tl.dot(x, wmu, out_dtype=tl.float32, input_precision="ieee")
    acc_sig = tl.dot(x2, ws2, out_dtype=tl.float32, input_precision="ieee")

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
        self._eps_state = None
        if torch.cuda.is_available():
            self._eps_state = torch.cuda.get_rng_state()
        self._eps_cache = {}
        self._BLOCK_N = max(16, triton.next_power_of_2(out_features))
        self._BLOCK_K = max(16, triton.next_power_of_2(in_features))

    def reset_parameters(self):
        init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        init.uniform_(self.w_log_sigma, self.log_sigma_prior_init - 0.1,
                      self.log_sigma_prior_init)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _get_eps(self, out_shape, device, dtype):
        key = (out_shape, device, dtype)
        e = self._eps_cache.get(key)
        if e is not None:
            return e
        if self._eps_state is not None and device.type == "cuda":
            saved = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self._eps_state)
            e = torch.randn(out_shape, device=device, dtype=dtype)
            torch.cuda.set_rng_state(saved)
        else:
            e = torch.randn(out_shape, device=device, dtype=dtype)
        e = e.view(-1, out_shape[-1])
        self._eps_cache[key] = e
        return e

    def forward(self, input):
        orig_shape = input.shape
        K = self.in_features
        N = self.out_features
        if not input.is_contiguous():
            input = input.contiguous()
        x = input.view(-1, K)
        M = x.shape[0]

        out_shape = tuple(orig_shape[:-1]) + (N,)
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        eps = self._get_eps(out_shape, x.device, x.dtype)

        BLOCK_M = max(16, triton.next_power_of_2(M))
        grid = (triton.cdiv(M, BLOCK_M),)

        _bayes_kernel[grid](
            x, self.w_mu, self.w_log_sigma,
            self.bias if self.bias is not None else x,
            eps, out, M, N, K,
            HAS_BIAS=self.bias is not None,
            BLOCK_M=BLOCK_M, BLOCK_N=self._BLOCK_N, BLOCK_K=self._BLOCK_K,
            num_warps=2,
        )
        return out.view(out_shape)
