import torch
import numpy as np
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mlp_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr, out_ptr,
                M, K, F1, F2, A,
                BLOCK_M: tl.constexpr, BK: tl.constexpr, BF1: tl.constexpr,
                BF2: tl.constexpr, BA: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    ok = tl.arange(0, BK)
    of1 = tl.arange(0, BF1)
    of2 = tl.arange(0, BF2)
    oa = tl.arange(0, BA)
    mmask = offs_m < M

    x = tl.load(x_ptr + offs_m[:, None] * K + ok[None, :],
                mask=mmask[:, None] & (ok[None, :] < K), other=0.0)

    # layer 1
    w1 = tl.load(w1_ptr + ok[:, None] + of1[None, :] * K,
                 mask=(ok[:, None] < K) & (of1[None, :] < F1), other=0.0)
    h1 = tl.dot(x, w1, out_dtype=tl.float32)
    b1 = tl.load(b1_ptr + of1, mask=of1 < F1, other=0.0)
    h1 = tl.maximum(h1 + b1[None, :], 0.0)

    # layer 2
    w2 = tl.load(w2_ptr + of1[:, None] + of2[None, :] * F1,
                 mask=(of1[:, None] < F1) & (of2[None, :] < F2), other=0.0)
    h2 = tl.dot(h1.to(tl.float32), w2, out_dtype=tl.float32)
    b2 = tl.load(b2_ptr + of2, mask=of2 < F2, other=0.0)
    h2 = tl.maximum(h2 + b2[None, :], 0.0)

    # layer 3 + tanh
    w3 = tl.load(w3_ptr + of2[:, None] + oa[None, :] * F2,
                 mask=(of2[:, None] < F2) & (oa[None, :] < A), other=0.0)
    o = tl.dot(h2.to(tl.float32), w3, out_dtype=tl.float32)
    b3 = tl.load(b3_ptr + oa, mask=oa < A, other=0.0)
    o = o + b3[None, :]
    o = tl.where(o > 0, 1.0, -1.0) * (1.0 - 2.0 / (tl.exp(2.0 * tl.abs(o)) + 1.0))

    tl.store(out_ptr + offs_m[:, None] * A + oa[None, :],
             o, mask=mmask[:, None] & (oa[None, :] < A))


class MADDPGActorVersion1New(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units, fc2_units):
        super().__init__()
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
        x = state.reshape(-1, orig_shape[-1]).contiguous().float()
        M, K = x.shape
        F1 = self.fc1.weight.shape[0]
        F2 = self.fc2.weight.shape[0]
        A = self.fc3.weight.shape[0]
        out = torch.empty((M, A), device=x.device, dtype=torch.float32)
        BM = min(triton.next_power_of_2(M), 256) if M > 0 else 16
        BK = max(16, triton.next_power_of_2(K))
        BF1 = max(16, triton.next_power_of_2(F1))
        BF2 = max(16, triton.next_power_of_2(F2))
        BA = max(16, triton.next_power_of_2(A))
        grid = (triton.cdiv(M, BM),)
        _mlp_kernel[grid](x, self.fc1.weight, self.fc1.bias,
                          self.fc2.weight, self.fc2.bias,
                          self.fc3.weight, self.fc3.bias, out,
                          M, K, F1, F2, A,
                          BLOCK_M=BM, BK=BK, BF1=BF1, BF2=BF2, BA=BA, num_warps=2)
        out_shape = orig_shape[:-1] + (A,)
        return out.reshape(out_shape)
