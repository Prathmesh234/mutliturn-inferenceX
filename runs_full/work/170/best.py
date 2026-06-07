import torch
import numpy as np
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_actor_kernel(a_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr,
                        c_ptr, M, K, H1, H2, A,
                        sa_m, sa_k, sw1_n, sw1_k, sw2_n, sw2_k, sw3_n, sw3_k,
                        sc_m, sc_n,
                        BM: tl.constexpr, BK: tl.constexpr, BH1: tl.constexpr,
                        BH2: tl.constexpr, BA: tl.constexpr):
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    offs_h1 = tl.arange(0, BH1)
    offs_h2 = tl.arange(0, BH2)
    offs_a = tl.arange(0, BA)

    a = tl.load(a_ptr + offs_m[:, None] * sa_m + offs_k[None, :] * sa_k,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
    w1 = tl.load(w1_ptr + offs_h1[:, None] * sw1_n + offs_k[None, :] * sw1_k,
                 mask=(offs_h1[:, None] < H1) & (offs_k[None, :] < K), other=0.0)
    x = tl.dot(a, tl.trans(w1))
    b1 = tl.load(b1_ptr + offs_h1, mask=offs_h1 < H1, other=0.0)
    x = tl.maximum(x + b1[None, :], 0.0)

    w2 = tl.load(w2_ptr + offs_h2[:, None] * sw2_n + offs_h1[None, :] * sw2_k,
                 mask=(offs_h2[:, None] < H2) & (offs_h1[None, :] < H1), other=0.0)
    y = tl.dot(x, tl.trans(w2))
    b2 = tl.load(b2_ptr + offs_h2, mask=offs_h2 < H2, other=0.0)
    y = tl.maximum(y + b2[None, :], 0.0)

    w3 = tl.load(w3_ptr + offs_a[:, None] * sw3_n + offs_h2[None, :] * sw3_k,
                 mask=(offs_a[:, None] < A) & (offs_h2[None, :] < H2), other=0.0)
    z = tl.dot(y, tl.trans(w3))
    b3 = tl.load(b3_ptr + offs_a, mask=offs_a < A, other=0.0)
    z = z + b3[None, :]
    z = (2.0 / (1.0 + tl.exp(-2.0 * z))) - 1.0

    tl.store(c_ptr + offs_m[:, None] * sc_m + offs_a[None, :] * sc_n, z,
             mask=(offs_m[:, None] < M) & (offs_a[None, :] < A))


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return -lim, lim


class DDPGActorVersion1New(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state):
        orig_shape = state.shape
        K = orig_shape[-1]
        a = state.reshape(-1, K).contiguous()
        M = a.shape[0]
        H1 = self.fc1.weight.shape[0]
        H2 = self.fc2.weight.shape[0]
        A = self.fc3.weight.shape[0]
        c = torch.empty((M, A), device=a.device, dtype=a.dtype)
        BM = triton.next_power_of_2(M)
        BK = max(16, triton.next_power_of_2(K))
        BH1 = max(16, triton.next_power_of_2(H1))
        BH2 = max(16, triton.next_power_of_2(H2))
        BA = max(16, triton.next_power_of_2(A))
        _fused_actor_kernel[(1,)](
            a, self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias,
            self.fc3.weight, self.fc3.bias, c, M, K, H1, H2, A,
            a.stride(0), a.stride(1),
            self.fc1.weight.stride(0), self.fc1.weight.stride(1),
            self.fc2.weight.stride(0), self.fc2.weight.stride(1),
            self.fc3.weight.stride(0), self.fc3.weight.stride(1),
            c.stride(0), c.stride(1),
            BM=BM, BK=BK, BH1=BH1, BH2=BH2, BA=BA, num_warps=4, num_stages=1)
        out_shape = orig_shape[:-1] + (A,)
        return c.reshape(out_shape)
