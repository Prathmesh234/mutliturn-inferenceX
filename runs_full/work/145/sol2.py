import torch
import torch.nn as nn
import numpy as np
import triton
import triton.language as tl


@triton.jit
def _fused_mlp(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, o_ptr,
               M,
               sx_m, sx_k,
               sw1_n, sw1_k, sw2_n, sw2_k, sw3_n, sw3_k,
               so_m, so_n,
               K1: tl.constexpr, N1: tl.constexpr, N2: tl.constexpr, N3: tl.constexpr,
               BK1: tl.constexpr, BN1: tl.constexpr, BN2: tl.constexpr, BN3: tl.constexpr,
               BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    # ---- layer 1: x[M,K1] @ w1[N1,K1]^T -> h1[M,N1], relu ----
    rk1 = tl.arange(0, BK1)
    x = tl.load(x_ptr + offs_m[:, None] * sx_m + rk1[None, :] * sx_k,
                mask=m_mask[:, None] & (rk1[None, :] < K1), other=0.0)
    rn1 = tl.arange(0, BN1)
    w1 = tl.load(w1_ptr + rn1[None, :] * sw1_n + rk1[:, None] * sw1_k,
                 mask=(rn1[None, :] < N1) & (rk1[:, None] < K1), other=0.0)
    h1 = tl.dot(x, w1)
    b1 = tl.load(b1_ptr + rn1, mask=rn1 < N1, other=0.0)
    h1 = h1 + b1[None, :]
    h1 = tl.where(h1 > 0, h1, 0.0)

    # ---- layer 2: h1[M,N1] @ w2[N2,N1]^T -> h2[M,N2], relu ----
    rn2 = tl.arange(0, BN2)
    w2 = tl.load(w2_ptr + rn2[None, :] * sw2_n + rn1[:, None] * sw2_k,
                 mask=(rn2[None, :] < N2) & (rn1[:, None] < N1), other=0.0)
    h2 = tl.dot(h1.to(w2.dtype), w2)
    b2 = tl.load(b2_ptr + rn2, mask=rn2 < N2, other=0.0)
    h2 = h2 + b2[None, :]
    h2 = tl.where(h2 > 0, h2, 0.0)

    # ---- layer 3: h2[M,N2] @ w3[N3,N2]^T -> out[M,N3], tanh ----
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
        out = torch.empty((M, N3), device=x.device, dtype=torch.float32)

        def pad(v):
            return max(16, triton.next_power_of_2(v))

        BLOCK_M = max(16, triton.next_power_of_2(M))
        grid = (triton.cdiv(M, BLOCK_M),)
        _fused_mlp[grid](x, self.fc1.weight, self.fc1.bias,
                         self.fc2.weight, self.fc2.bias,
                         self.fc3.weight, self.fc3.bias, out,
                         M,
                         x.stride(0), x.stride(1),
                         self.fc1.weight.stride(0), self.fc1.weight.stride(1),
                         self.fc2.weight.stride(0), self.fc2.weight.stride(1),
                         self.fc3.weight.stride(0), self.fc3.weight.stride(1),
                         out.stride(0), out.stride(1),
                         K1=K1, N1=N1, N2=N2, N3=N3,
                         BK1=pad(K1), BN1=pad(N1), BN2=pad(N2), BN3=pad(N3),
                         BLOCK_M=BLOCK_M, num_warps=8, num_stages=2)
        return out.reshape(*orig_shape[:-1], N3)
