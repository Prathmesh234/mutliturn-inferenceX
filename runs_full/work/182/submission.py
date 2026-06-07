import torch
import numpy as np
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(state_ptr, act_ptr,
                  w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_ptr,
                  out_ptr, s_sm, s_am,
                  KS: tl.constexpr, KA: tl.constexpr,
                  H1: tl.constexpr, H2: tl.constexpr,
                  BKS: tl.constexpr, BKA: tl.constexpr,
                  BH1: tl.constexpr, BH2: tl.constexpr):
    pid = tl.program_id(0)
    offs_ks = tl.arange(0, BKS)
    offs_ka = tl.arange(0, BKA)
    offs_h1 = tl.arange(0, BH1)
    offs_h2 = tl.arange(0, BH2)

    state = tl.load(state_ptr + pid * s_sm + offs_ks, mask=offs_ks < KS, other=0.0).to(tl.float32)
    w1 = tl.load(w1_ptr + offs_h1[:, None] * KS + offs_ks[None, :],
                 mask=(offs_h1[:, None] < H1) & (offs_ks[None, :] < KS), other=0.0).to(tl.float32)
    xs = tl.sum(state[None, :] * w1, axis=1)
    b1 = tl.load(b1_ptr + offs_h1, mask=offs_h1 < H1, other=0.0).to(tl.float32)
    xs = tl.maximum(xs + b1, 0.0)

    act = tl.load(act_ptr + pid * s_am + offs_ka, mask=offs_ka < KA, other=0.0).to(tl.float32)
    w2a = tl.load(w2_ptr + offs_h2[:, None] * (H1 + KA) + offs_h1[None, :],
                  mask=(offs_h2[:, None] < H2) & (offs_h1[None, :] < H1), other=0.0).to(tl.float32)
    w2b = tl.load(w2_ptr + offs_h2[:, None] * (H1 + KA) + (H1 + offs_ka[None, :]),
                  mask=(offs_h2[:, None] < H2) & (offs_ka[None, :] < KA), other=0.0).to(tl.float32)
    x = tl.sum(xs[None, :] * w2a, axis=1) + tl.sum(act[None, :] * w2b, axis=1)
    b2 = tl.load(b2_ptr + offs_h2, mask=offs_h2 < H2, other=0.0).to(tl.float32)
    x = tl.maximum(x + b2, 0.0)

    w3 = tl.load(w3_ptr + offs_h2, mask=offs_h2 < H2, other=0.0).to(tl.float32)
    b3 = tl.load(b3_ptr).to(tl.float32)
    out = tl.sum(x * w3) + b3
    tl.store(out_ptr + pid, out)


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
        KS = self.fcs1.weight.shape[1]
        KA = action.shape[1]
        H1 = self.fcs1.weight.shape[0]
        H2 = self.fc2.weight.shape[0]
        out = torch.empty((M, 1), device=state.device, dtype=torch.float32)
        _fused_kernel[(M,)](
            state, action,
            self.fcs1.weight, self.fcs1.bias,
            self.fc2.weight, self.fc2.bias,
            self.fc3.weight, self.fc3.bias,
            out, state.stride(0), action.stride(0),
            KS=KS, KA=KA, H1=H1, H2=H2,
            BKS=_next_pow2(KS), BKA=_next_pow2(KA),
            BH1=_next_pow2(H1), BH2=_next_pow2(H2),
            num_warps=1)
        return out
