import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(s_ptr, a_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, out_ptr,
                  M, NOBS, NACT, H,
                  BLOCK_M: tl.constexpr, BLOCK_IN: tl.constexpr, BLOCK_H: tl.constexpr):
    offs_m = tl.arange(0, BLOCK_M)
    offs_in = tl.arange(0, BLOCK_IN)
    offs_h = tl.arange(0, BLOCK_H)
    K = NOBS + NACT

    s = tl.load(s_ptr + offs_m[:, None] * NOBS + offs_in[None, :],
                mask=(offs_m[:, None] < M) & (offs_in[None, :] < NOBS), other=0.0)
    a = tl.load(a_ptr + offs_m[:, None] * NACT + (offs_in[None, :] - NOBS),
                mask=(offs_m[:, None] < M) & (offs_in[None, :] >= NOBS) & (offs_in[None, :] < K), other=0.0)
    x = tl.where(offs_in[None, :] < NOBS, s, a)

    w1 = tl.load(w1_ptr + offs_h[None, :] * K + offs_in[:, None],
                 mask=(offs_h[None, :] < H) & (offs_in[:, None] < K), other=0.0)
    h1 = tl.dot(x, w1)
    b1 = tl.load(b1_ptr + offs_h, mask=offs_h < H, other=0.0)
    h1 = tl.maximum(h1 + b1[None, :], 0.0)

    w2 = tl.load(w2_ptr + offs_h[None, :] * H + offs_h[:, None],
                 mask=(offs_h[None, :] < H) & (offs_h[:, None] < H), other=0.0)
    h2 = tl.dot(h1, w2)
    b2 = tl.load(b2_ptr + offs_h, mask=offs_h < H, other=0.0)
    h2 = tl.maximum(h2 + b2[None, :], 0.0)

    w3 = tl.load(w3_ptr + offs_h, mask=offs_h < H, other=0.0)
    out = tl.sum(h2 * w3[None, :], axis=1)
    b3 = tl.load(b3_ptr)
    out = out + b3
    tl.store(out_ptr + offs_m, out, mask=offs_m < M)


class CriticNew(nn.Module):
    def __init__(self, n_obs, output_dim, hidden_size, init_w=0.003):
        super().__init__()
        self.linear1 = nn.Linear(n_obs + output_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        self.hidden_size = hidden_size

    def forward(self, state, action):
        M = state.shape[0]
        NOBS = state.shape[1]
        NACT = action.shape[1]
        H = self.hidden_size
        out = torch.empty((M, 1), device=state.device, dtype=state.dtype)
        BLOCK_M = max(16, triton.next_power_of_2(M))
        BLOCK_IN = max(16, triton.next_power_of_2(NOBS + NACT))
        BLOCK_H = max(16, triton.next_power_of_2(H))
        _fused_kernel[(1,)](state, action,
                            self.linear1.weight, self.linear1.bias,
                            self.linear2.weight, self.linear2.bias,
                            self.linear3.weight, self.linear3.bias,
                            out, M, NOBS, NACT, H,
                            BLOCK_M=BLOCK_M, BLOCK_IN=BLOCK_IN, BLOCK_H=BLOCK_H,
                            num_warps=1, num_stages=2)
        return out
