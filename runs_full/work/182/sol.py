import torch
import numpy as np
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_kernel(x_ptr, w_ptr, b_ptr, out_ptr, M, N, K,
                   stride_xm, stride_xk, stride_wn, stride_wk,
                   BN: tl.constexpr, BK: tl.constexpr, APPLY_RELU: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    x = tl.load(x_ptr + pid_m * stride_xm + offs_k * stride_xk,
                mask=offs_k < K, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0).to(tl.float32)
    acc = tl.sum(x[None, :] * w, axis=1)
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += b
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)
    tl.store(out_ptr + pid_m * N + offs_n, acc, mask=offs_n < N)


@triton.jit
def _fc2_kernel(xs_ptr, act_ptr, w_ptr, b_ptr, out_ptr, M, N, K1, K2,
                stride_xm, stride_am, stride_wn, stride_wk,
                BN: tl.constexpr, BK1: tl.constexpr, BK2: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k1 = tl.arange(0, BK1)
    offs_k2 = tl.arange(0, BK2)

    xs = tl.load(xs_ptr + pid_m * stride_xm + offs_k1,
                 mask=offs_k1 < K1, other=0.0).to(tl.float32)
    act = tl.load(act_ptr + pid_m * stride_am + offs_k2,
                  mask=offs_k2 < K2, other=0.0).to(tl.float32)

    w1 = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k1[None, :] * stride_wk,
                 mask=(offs_n[:, None] < N) & (offs_k1[None, :] < K1), other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + offs_n[:, None] * stride_wn + (K1 + offs_k2[None, :]) * stride_wk,
                 mask=(offs_n[:, None] < N) & (offs_k2[None, :] < K2), other=0.0).to(tl.float32)

    acc = tl.sum(xs[None, :] * w1, axis=1) + tl.sum(act[None, :] * w2, axis=1)
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += b
    acc = tl.maximum(acc, 0.0)
    tl.store(out_ptr + pid_m * N + offs_n, acc, mask=offs_n < N)


def _next_pow2(x):
    return 1 << (max(x, 1) - 1).bit_length()


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return -lim, lim


class DDPGCriticVersion1New(nn.Module):

    def __init__(self, state_size, action_size, seed, fcs1_units=128,
                 fc2_units=128):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state, action):
        state = state.contiguous()
        action = action.contiguous()
        M = state.shape[0]

        # fcs1: relu(state @ W1^T + b1)
        N1 = self.fcs1.weight.shape[0]
        K1 = self.fcs1.weight.shape[1]
        xs = torch.empty((M, N1), device=state.device, dtype=torch.float32)
        BN1 = _next_pow2(N1)
        BK = _next_pow2(K1)
        _linear_kernel[(M, triton.cdiv(N1, BN1))](
            state, self.fcs1.weight, self.fcs1.bias, xs, M, N1, K1,
            state.stride(0), state.stride(1),
            self.fcs1.weight.stride(0), self.fcs1.weight.stride(1),
            BN=BN1, BK=BK, APPLY_RELU=True, num_warps=4)

        # fc2: relu(cat(xs, action) @ W2^T + b2)
        N2 = self.fc2.weight.shape[0]
        Ka = action.shape[1]
        x = torch.empty((M, N2), device=state.device, dtype=torch.float32)
        BN2 = _next_pow2(N2)
        _fc2_kernel[(M, triton.cdiv(N2, BN2))](
            xs, action, self.fc2.weight, self.fc2.bias, x, M, N2, N1, Ka,
            xs.stride(0), action.stride(0),
            self.fc2.weight.stride(0), self.fc2.weight.stride(1),
            BN=BN2, BK1=_next_pow2(N1), BK2=_next_pow2(Ka), num_warps=4)

        # fc3: x @ W3^T + b3
        N3 = self.fc3.weight.shape[0]
        K3 = self.fc3.weight.shape[1]
        out = torch.empty((M, N3), device=state.device, dtype=torch.float32)
        _linear_kernel[(M, 1)](
            x, self.fc3.weight, self.fc3.bias, out, M, N3, K3,
            x.stride(0), x.stride(1),
            self.fc3.weight.stride(0), self.fc3.weight.stride(1),
            BN=_next_pow2(N3), BK=_next_pow2(K3), APPLY_RELU=False, num_warps=4)

        return out
