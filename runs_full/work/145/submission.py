import torch
import torch.nn as nn
import numpy as np
import triton
import triton.language as tl


@triton.jit
def _linear_act(x_ptr, w_ptr, b_ptr, o_ptr,
                M, N, K,
                stride_xm, stride_xk,
                stride_wn, stride_wk,
                stride_om, stride_on,
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
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
                    mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
        w = tl.load(w_ptr + offs_n[None, :] * stride_wn + k[:, None] * stride_wk,
                    mask=(offs_n[None, :] < N) & (k[:, None] < K), other=0.0)
        acc += tl.dot(x, w)
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b[None, :]
    if ACT == 1:
        acc = tl.where(acc > 0, acc, 0.0)
    elif ACT == 2:
        acc = (2.0 / (1.0 + tl.exp(-2.0 * acc))) - 1.0
    tl.store(o_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _launch(x, w, b, act):
    M, K = x.shape
    N = w.shape[0]
    o = torch.empty((M, N), device=x.device, dtype=torch.float32)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 16
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _linear_act[grid](x, w, b, o, M, N, K,
                      x.stride(0), x.stride(1),
                      w.stride(0), w.stride(1),
                      o.stride(0), o.stride(1),
                      ACT=act, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                      num_warps=4, num_stages=3)
    return o


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
        h1 = _launch(x, self.fc1.weight, self.fc1.bias, 1)
        h2 = _launch(h1, self.fc2.weight, self.fc2.bias, 1)
        out = _launch(h2, self.fc3.weight, self.fc3.bias, 2)
        return out.reshape(*orig_shape[:-1], out.shape[-1])
