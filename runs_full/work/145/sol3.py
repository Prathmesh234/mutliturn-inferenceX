import torch
import torch.nn as nn
import numpy as np
import triton
import triton.language as tl


@triton.jit
def _linear_act(x_ptr, w_ptr, b_ptr, o_ptr,
                M, N, K,
                sx_m, sx_k, sw_n, sw_k, so_m, so_n,
                ACT: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        x = tl.load(x_ptr + offs_m[:, None] * sx_m + k[None, :] * sx_k,
                    mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
        w = tl.load(w_ptr + offs_n[None, :] * sw_n + k[:, None] * sw_k,
                    mask=(offs_n[None, :] < N) & (k[:, None] < K), other=0.0)
        acc += tl.dot(x, w)
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b[None, :]
    if ACT == 1:
        acc = tl.where(acc > 0, acc, 0.0)
    tl.store(o_ptr + offs_m[:, None] * so_m + offs_n[None, :] * so_n,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _fc2_fc3(h1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, o_ptr,
             M,
             sh_m, sh_k, sw2_n, sw2_k, sw3_n, sw3_k, so_m, so_n,
             N1: tl.constexpr, N2: tl.constexpr, N3: tl.constexpr,
             BK: tl.constexpr, BN2: tl.constexpr, BN3: tl.constexpr,
             BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    rn2 = tl.arange(0, BN2)
    acc = tl.zeros((BLOCK_M, BN2), dtype=tl.float32)
    for k0 in range(0, N1, BK):
        k = k0 + tl.arange(0, BK)
        h = tl.load(h1_ptr + offs_m[:, None] * sh_m + k[None, :] * sh_k,
                    mask=m_mask[:, None] & (k[None, :] < N1), other=0.0)
        w2 = tl.load(w2_ptr + rn2[None, :] * sw2_n + k[:, None] * sw2_k,
                     mask=(rn2[None, :] < N2) & (k[:, None] < N1), other=0.0)
        acc += tl.dot(h, w2)
    b2 = tl.load(b2_ptr + rn2, mask=rn2 < N2, other=0.0)
    h2 = acc + b2[None, :]
    h2 = tl.where(h2 > 0, h2, 0.0)

    rn3 = tl.arange(0, BN3)
    w3 = tl.load(w3_ptr + rn3[None, :] * sw3_n + rn2[:, None] * sw3_k,
                 mask=(rn3[None, :] < N3) & (rn2[:, None] < N2), other=0.0)
    out = tl.dot(h2.to(w3.dtype), w3)
    b3 = tl.load(b3_ptr + rn3, mask=rn3 < N3, other=0.0)
    out = out + b3[None, :]
    out = (2.0 / (1.0 + tl.exp(-2.0 * out))) - 1.0
    tl.store(o_ptr + offs_m[:, None] * so_m + rn3[None, :] * so_n,
             out, mask=m_mask[:, None] & (rn3[None, :] < N3))


class ActorNew(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        super(ActorNew, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        def hidden_init(layer):
            fan_in = layer.weight.data.size()[0]
            lim = 1.0 / np.sqrt(fan_in)
            return -lim, lim
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state):
        orig_shape = state.shape
        x = state.reshape(-1, orig_shape[-1]).contiguous().to(torch.float32)
        M = x.shape[0]
        K1 = self.fc1.weight.shape[1]
        N1 = self.fc1.weight.shape[0]
        N2 = self.fc2.weight.shape[0]
        N3 = self.fc3.weight.shape[0]

        def pad(v):
            return max(16, triton.next_power_of_2(v))

        BLOCK_M = max(16, triton.next_power_of_2(M))

        h1 = torch.empty((M, N1), device=x.device, dtype=torch.float32)
        _linear_act[(triton.cdiv(M, BLOCK_M), triton.cdiv(N1, 64))](
            x, self.fc1.weight, self.fc1.bias, h1, M, N1, K1,
            x.stride(0), x.stride(1), self.fc1.weight.stride(0), self.fc1.weight.stride(1),
            h1.stride(0), h1.stride(1), ACT=1,
            BLOCK_M=BLOCK_M, BLOCK_N=64, BLOCK_K=pad(K1), num_warps=4, num_stages=3)

        out = torch.empty((M, N3), device=x.device, dtype=torch.float32)
        _fc2_fc3[(triton.cdiv(M, BLOCK_M),)](
            h1, self.fc2.weight, self.fc2.bias, self.fc3.weight, self.fc3.bias, out, M,
            h1.stride(0), h1.stride(1), self.fc2.weight.stride(0), self.fc2.weight.stride(1),
            self.fc3.weight.stride(0), self.fc3.weight.stride(1), out.stride(0), out.stride(1),
            N1=N1, N2=N2, N3=N3, BK=32, BN2=pad(N2), BN3=pad(N3),
            BLOCK_M=BLOCK_M, num_warps=4, num_stages=2)
        return out.reshape(*orig_shape[:-1], N3)
