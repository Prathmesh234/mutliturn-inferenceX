import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _fc1_kernel(x_ptr, w_ptr, b_ptr, y_ptr, M, K, H,
                BLOCK_M: tl.constexpr, H_C: tl.constexpr, KC: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = tl.arange(0, H_C)
    offs_k = tl.arange(0, KC)
    m_mask = offs_m < M
    x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=m_mask[:, None] & (offs_k[None, :] < K), other=0.0)
    w = tl.load(w_ptr + offs_h[None, :] * K + offs_k[:, None],
                mask=(offs_h[None, :] < H) & (offs_k[:, None] < K), other=0.0)
    acc = tl.dot(x, w)
    b = tl.load(b_ptr + offs_h, mask=offs_h < H, other=0.0)
    acc = tl.maximum(acc + b[None, :], 0.0)
    tl.store(y_ptr + offs_m[:, None] * H + offs_h[None, :], acc,
             mask=m_mask[:, None] & (offs_h[None, :] < H))


@triton.jit
def _fc23_kernel(h1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, y_ptr, M, H, N_out,
                 BLOCK_M: tl.constexpr, H_C: tl.constexpr, NOUT_C: tl.constexpr,
                 BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = tl.arange(0, H_C)
    offs_no = tl.arange(0, NOUT_C)
    m_mask = offs_m < M
    acc2 = tl.zeros((BLOCK_M, H_C), dtype=tl.float32)
    for k0 in range(0, H, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        h1b = tl.load(h1_ptr + offs_m[:, None] * H + kk[None, :],
                      mask=m_mask[:, None] & (kk[None, :] < H), other=0.0)
        w2b = tl.load(w2_ptr + offs_h[None, :] * H + kk[:, None],
                      mask=(offs_h[None, :] < H) & (kk[:, None] < H), other=0.0)
        acc2 += tl.dot(h1b, w2b)
    b2 = tl.load(b2_ptr + offs_h, mask=offs_h < H, other=0.0)
    h2 = tl.maximum(acc2 + b2[None, :], 0.0)
    w3 = tl.load(w3_ptr + offs_no[None, :] * H + offs_h[:, None],
                 mask=(offs_no[None, :] < N_out) & (offs_h[:, None] < H), other=0.0)
    o = tl.dot(h2, w3)
    b3 = tl.load(b3_ptr + offs_no, mask=offs_no < N_out, other=0.0)
    o = o + b3[None, :]
    o = 2.0 * tl.sigmoid(2.0 * o) - 1.0
    tl.store(y_ptr + offs_m[:, None] * N_out + offs_no[None, :], o,
             mask=m_mask[:, None] & (offs_no[None, :] < N_out))


class Policy_NetNew(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(Policy_NetNew, self).__init__()
        self.fc1 = nn.Linear(observation_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, observation):
        orig = observation.shape
        K = orig[-1]
        x = observation.reshape(-1, K).contiguous()
        M = x.shape[0]
        H = 256
        N_out = self.fc3.weight.shape[0]
        KC = max(16, triton.next_power_of_2(K))
        NOUT_C = max(16, triton.next_power_of_2(N_out))
        BLOCK_M = 32
        h1 = torch.empty((M, H), device=x.device, dtype=x.dtype)
        y = torch.empty((M, N_out), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, BLOCK_M),)
        _fc1_kernel[grid](x, self.fc1.weight, self.fc1.bias, h1, M, K, H,
                          BLOCK_M=BLOCK_M, H_C=H, KC=KC, num_warps=2, num_stages=2)
        _fc23_kernel[grid](h1, self.fc2.weight, self.fc2.bias,
                           self.fc3.weight, self.fc3.bias, y, M, H, N_out,
                           BLOCK_M=BLOCK_M, H_C=H, NOUT_C=NOUT_C, BLOCK_K=16,
                           num_warps=2, num_stages=2)
        return y.reshape(*orig[:-1], N_out)
