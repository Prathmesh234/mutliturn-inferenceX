import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(s_ptr, a_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, out_ptr,
                  M, OBS, ACT,
                  ss_m, ss_k, sa_m, sa_k,
                  BLOCK_M: tl.constexpr, K0P: tl.constexpr, H: tl.constexpr, BN: tl.constexpr):
    offs_m = tl.arange(0, BLOCK_M)
    offs_h = tl.arange(0, H)
    offs_k0 = tl.arange(0, K0P)
    offs_n = tl.arange(0, 16)
    offs_bn = tl.arange(0, BN)
    K0 = OBS + ACT
    xs = tl.load(s_ptr + offs_m[:, None] * ss_m + offs_k0[None, :] * ss_k,
                 mask=(offs_m[:, None] < M) & (offs_k0[None, :] < OBS), other=0.0)
    xa = tl.load(a_ptr + offs_m[:, None] * sa_m + (offs_k0[None, :] - OBS) * sa_k,
                 mask=(offs_m[:, None] < M) & (offs_k0[None, :] >= OBS) & (offs_k0[None, :] < K0), other=0.0)
    x = xs + xa
    w1 = tl.load(w1_ptr + offs_h[:, None] * K0 + offs_k0[None, :],
                 mask=offs_k0[None, :] < K0, other=0.0)
    h = tl.dot(x, tl.trans(w1), allow_tf32=False)
    b1 = tl.load(b1_ptr + offs_h)
    h = tl.maximum(h + b1[None, :], 0.0)
    acc = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    for n0 in range(0, H, BN):
        nn_ = n0 + offs_bn
        w2 = tl.load(w2_ptr + nn_[:, None] * H + offs_h[None, :])
        h2 = tl.dot(h, tl.trans(w2), allow_tf32=False)
        b2 = tl.load(b2_ptr + nn_)
        h2 = tl.maximum(h2 + b2[None, :], 0.0)
        w3 = tl.load(w3_ptr + offs_n[:, None] * H + nn_[None, :],
                     mask=offs_n[:, None] < 1, other=0.0)
        acc += tl.dot(h2, tl.trans(w3), allow_tf32=False)
    b3 = tl.load(b3_ptr + offs_n, mask=offs_n < 1, other=0.0)
    o = acc + b3[None, :]
    tl.store(out_ptr + offs_m[:, None] * 1 + offs_n[None, :] * 1,
             o, mask=(offs_m[:, None] < M) & (offs_n[None, :] < 1))


class Value_NetNew(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(observation_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        M, OBS = state.shape
        ACT = action.shape[1]
        out = torch.empty((M, 1), device=state.device, dtype=state.dtype)
        BLOCK_M = max(triton.next_power_of_2(M), 16)
        K0P = max(triton.next_power_of_2(OBS + ACT), 16)
        _fused_kernel[(1,)](state, action, self.fc1.weight, self.fc1.bias,
                            self.fc2.weight, self.fc2.bias,
                            self.fc3.weight, self.fc3.bias, out,
                            M, OBS, ACT,
                            state.stride(0), state.stride(1),
                            action.stride(0), action.stride(1),
                            BLOCK_M=BLOCK_M, K0P=K0P, H=256, BN=128,
                            num_warps=4, num_stages=2)
        return out
