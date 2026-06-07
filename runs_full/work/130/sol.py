import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _linear_kernel(x_ptr, w_ptr, b_ptr, y_ptr, M, N, K,
                   stride_xm, stride_xk, stride_wn, stride_wk,
                   stride_ym, stride_yn,
                   ACT: tl.constexpr, BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        acc += tl.dot(x, tl.trans(w))
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b[None, :]
    if ACT == 1:
        acc = tl.where(acc > 0, acc, 0.0)
    elif ACT == 2:
        acc = (2.0 / (1.0 + tl.exp(-2.0 * acc))) - 1.0
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _linear(x, weight, bias, act):
    M, K = x.shape
    N = weight.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)
    BLOCK_M = 16
    BLOCK_N = 256 if N >= 16 else 16
    BLOCK_K = 256
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _linear_kernel[grid](x, weight, bias, y, M, N, K,
                         x.stride(0), x.stride(1), weight.stride(0), weight.stride(1),
                         y.stride(0), y.stride(1),
                         act, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=4)
    return y


class Value_Net(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(Value_Net, self).__init__()
        self.fc1 = nn.Linear(observation_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1).contiguous()
        x = _linear(x, self.fc1.weight, self.fc1.bias, 1)
        x = _linear(x, self.fc2.weight, self.fc2.bias, 1)
        return _linear(x, self.fc3.weight, self.fc3.bias, 0)


class Policy_Net(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(Policy_Net, self).__init__()
        self.fc1 = nn.Linear(observation_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, observation):
        x = _linear(observation.contiguous(), self.fc1.weight, self.fc1.bias, 1)
        x = _linear(x, self.fc2.weight, self.fc2.bias, 1)
        return _linear(x, self.fc3.weight, self.fc3.bias, 2)


class DDPGNew(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(DDPGNew, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.actor = Policy_Net(self.observation_dim, self.action_dim)
        self.critic = Value_Net(self.observation_dim, self.action_dim)

    def forward(self, state):
        action = self.actor(state)
        value = self.critic(state, action)
        return action, value
