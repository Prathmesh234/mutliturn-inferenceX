import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mlp_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, out_ptr,
                M, K0, N1, N3,
                BM: tl.constexpr, BK0: tl.constexpr, BN1: tl.constexpr, BN3: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    m_mask = offs_m < M

    k0 = tl.arange(0, BK0)
    n1 = tl.arange(0, BN1)
    n3 = tl.arange(0, BN3)
    k0_mask = k0 < K0
    n1_mask = n1 < N1
    n3_mask = n3 < N3

    x = tl.load(x_ptr + offs_m[:, None] * K0 + k0[None, :],
                mask=m_mask[:, None] & k0_mask[None, :], other=0.0)

    w1 = tl.load(w1_ptr + n1[:, None] * K0 + k0[None, :],
                 mask=n1_mask[:, None] & k0_mask[None, :], other=0.0)
    b1 = tl.load(b1_ptr + n1, mask=n1_mask, other=0.0)
    h1 = tl.sum(x[:, None, :] * w1[None, :, :], axis=2) + b1[None, :]
    h1 = tl.maximum(h1, 0.0)

    w2 = tl.load(w2_ptr + n1[:, None] * N1 + n1[None, :],
                 mask=n1_mask[:, None] & n1_mask[None, :], other=0.0)
    b2 = tl.load(b2_ptr + n1, mask=n1_mask, other=0.0)
    h2 = tl.sum(h1[:, None, :] * w2[None, :, :], axis=2) + b2[None, :]
    h2 = tl.maximum(h2, 0.0)

    w3 = tl.load(w3_ptr + n3[:, None] * N1 + n1[None, :],
                 mask=n3_mask[:, None] & n1_mask[None, :], other=0.0)
    b3 = tl.load(b3_ptr + n3, mask=n3_mask, other=0.0)
    out = tl.sum(h2[:, None, :] * w3[None, :, :], axis=2) + b3[None, :]

    tl.store(out_ptr + offs_m[:, None] * N3 + n3[None, :],
             out, mask=m_mask[:, None] & n3_mask[None, :])


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


class NetNew(nn.Module):
    def __init__(self, n_obs, n_mid, n_action):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_action)

    def forward(self, x):
        K0 = self.fc1.in_features
        N1 = self.fc1.out_features
        N3 = self.fc3.out_features
        orig_shape = x.shape
        x2d = x.reshape(-1, K0)
        M = x2d.shape[0]
        out = torch.empty((M, N3), device=x.device, dtype=x.dtype)

        BM = 128
        BK0 = _next_pow2(K0)
        BN1 = _next_pow2(N1)
        BN3 = _next_pow2(N3)
        grid = (triton.cdiv(M, BM),)
        _mlp_kernel[grid](x2d, self.fc1.weight, self.fc1.bias,
                          self.fc2.weight, self.fc2.bias,
                          self.fc3.weight, self.fc3.bias, out,
                          M, K0, N1, N3,
                          BM=BM, BK0=BK0, BN1=BN1, BN3=BN3, num_warps=1)
        return out.reshape(*orig_shape[:-1], N3)
