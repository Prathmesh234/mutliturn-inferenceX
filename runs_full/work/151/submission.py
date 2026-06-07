import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_kernel(x_ptr, w_ptr, b_ptr, o_ptr, M, N, K,
                   sxm, sxk, swn, swk, som, son,
                   APPLY_RELU: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    om = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    ok = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kk = k * BLOCK_K + ok
        x = tl.load(x_ptr + om[:, None] * sxm + kk[None, :] * sxk,
                    mask=(om[:, None] < M) & (kk[None, :] < K), other=0.0)
        w = tl.load(w_ptr + on[:, None] * swn + kk[None, :] * swk,
                    mask=(on[:, None] < N) & (kk[None, :] < K), other=0.0)
        acc += tl.dot(x, tl.trans(w))
    b = tl.load(b_ptr + on, mask=on < N, other=0.0)
    acc += b[None, :]
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)
    tl.store(o_ptr + om[:, None] * som + on[None, :] * son, acc,
             mask=(om[:, None] < M) & (on[None, :] < N))


def linear(x, w, b, relu):
    M, K = x.shape
    N = w.shape[0]
    o = torch.empty((M, N), device=x.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 256, 16
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _linear_kernel[grid](x, w, b, o, M, N, K,
                         x.stride(0), x.stride(1), w.stride(0), w.stride(1),
                         o.stride(0), o.stride(1), APPLY_RELU=relu,
                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4)
    return o


class CriticNew(nn.Module):
    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        super(CriticNew, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        import numpy as np
        def hidden_init(layer):
            fan_in = layer.weight.data.size()[0]
            lim = 1.0 / np.sqrt(fan_in)
            return -lim, lim
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state, action):
        xs = linear(state.contiguous(), self.fcs1.weight, self.fcs1.bias, True)
        x = torch.cat((xs, action), dim=1).contiguous()
        x = linear(x, self.fc2.weight, self.fc2.bias, True)
        return linear(x, self.fc3.weight, self.fc3.bias, False)
