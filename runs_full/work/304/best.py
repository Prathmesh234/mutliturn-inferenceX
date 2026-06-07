import torch
import numpy as np
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _exp_normal_kernel(dist_ptr, mu_ptr, beta_ptr, out_ptr, N, n_rbf,
                       BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    M = N * n_rbf
    mask = offs < M
    dist_idx = offs // n_rbf
    rbf_idx = offs % n_rbf
    d = tl.load(dist_ptr + dist_idx, mask=mask, other=0.0)
    mu = tl.load(mu_ptr + rbf_idx, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + rbf_idx, mask=mask, other=0.0)
    ed = tl.exp(-d)
    arg = beta * (ed - mu) * (ed - mu)
    out = tl.exp(-arg)
    tl.store(out_ptr + offs, out, mask=mask)


class ExpNormalBasisNew(nn.Module):

    def __init__(self, n_rbf, cutoff, learnable_mu, learnable_beta):
        super().__init__()
        self.mu = torch.linspace(np.exp(-cutoff), 1, n_rbf)
        init_beta = (2 / n_rbf * (1 - np.exp(-cutoff))) ** -2
        self.beta = torch.ones_like(self.mu) * init_beta
        if learnable_mu:
            self.mu = nn.Parameter(self.mu)
        if learnable_beta:
            self.beta = nn.Parameter(self.beta)
        self.cutoff = cutoff

    def forward(self, dist):
        n_rbf = self.mu.shape[0]
        out = torch.empty(dist.shape + (n_rbf,), device=dist.device,
                          dtype=dist.dtype)
        N = dist.numel()
        M = N * n_rbf
        dist_c = dist.contiguous()
        mu = self.mu.contiguous()
        beta = self.beta.contiguous()
        BLOCK = 1024
        grid = (triton.cdiv(M, BLOCK),)
        _exp_normal_kernel[grid](dist_c, mu, beta, out, N, n_rbf,
                                 BLOCK=BLOCK, num_warps=4)
        return out
