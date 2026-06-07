import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _ddpg_kernel(x_ptr,
                 w1a, b1a, w2a, b2a, w3a, b3a,
                 w1c, b1c, w2c, b2c, w3c, b3c,
                 yact_ptr, yval_ptr,
                 M, OBS, ADIM,
                 sx0, sx1,
                 s1a0, s1a1, s2a0, s2a1, s3a0, s3a1,
                 s1c0, s1c1, s2c0, s2c1, s3c0, s3c1,
                 sya0, sya1, syv0, syv1,
                 H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
                 BLOCK_J: tl.constexpr, BLOCK_OUT: tl.constexpr):
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_all = tl.arange(0, H)
    offs_out = tl.arange(0, BLOCK_OUT)

    xa = tl.load(x_ptr + offs_m[:, None] * sx0 + offs_k[None, :] * sx1,
                 mask=(offs_m[:, None] < M) & (offs_k[None, :] < OBS), other=0.0)
    acc2 = tl.zeros((BLOCK_M, H), tl.float32)
    for j in range(0, H, BLOCK_J):
        offs_j = j + tl.arange(0, BLOCK_J)
        w1t = tl.load(w1a + offs_j[:, None] * s1a0 + offs_k[None, :] * s1a1,
                      mask=(offs_j[:, None] < H) & (offs_k[None, :] < OBS), other=0.0)
        h1t = tl.dot(xa, tl.trans(w1t))
        h1t += tl.load(b1a + offs_j, mask=offs_j < H, other=0.0)[None, :]
        h1t = tl.where(h1t > 0, h1t, 0.0)
        w2t = tl.load(w2a + offs_all[:, None] * s2a0 + offs_j[None, :] * s2a1)
        acc2 += tl.dot(h1t, tl.trans(w2t))
    acc2 += tl.load(b2a + offs_all)[None, :]
    h2 = tl.where(acc2 > 0, acc2, 0.0)
    w3t = tl.load(w3a + offs_out[:, None] * s3a0 + offs_all[None, :] * s3a1,
                  mask=offs_out[:, None] < ADIM, other=0.0)
    acc3 = tl.dot(h2, tl.trans(w3t))
    acc3 += tl.load(b3a + offs_out, mask=offs_out < ADIM, other=0.0)[None, :]
    action = (2.0 / (1.0 + tl.exp(-2.0 * acc3))) - 1.0
    tl.store(yact_ptr + offs_m[:, None] * sya0 + offs_out[None, :] * sya1, action,
             mask=(offs_m[:, None] < M) & (offs_out[None, :] < ADIM))

    xs = tl.load(x_ptr + offs_m[:, None] * sx0 + offs_k[None, :] * sx1,
                 mask=(offs_m[:, None] < M) & (offs_k[None, :] < OBS), other=0.0)
    accc = tl.zeros((BLOCK_M, H), tl.float32)
    for j in range(0, H, BLOCK_J):
        offs_j = j + tl.arange(0, BLOCK_J)
        w1s = tl.load(w1c + offs_j[:, None] * s1c0 + offs_k[None, :] * s1c1,
                      mask=(offs_j[:, None] < H) & (offs_k[None, :] < OBS), other=0.0)
        h1t = tl.dot(xs, tl.trans(w1s))
        w1ac = tl.load(w1c + offs_j[:, None] * s1c0 + (OBS + offs_out[None, :]) * s1c1,
                       mask=(offs_j[:, None] < H) & (offs_out[None, :] < ADIM), other=0.0)
        h1t += tl.dot(action, tl.trans(w1ac))
        h1t += tl.load(b1c + offs_j, mask=offs_j < H, other=0.0)[None, :]
        h1t = tl.where(h1t > 0, h1t, 0.0)
        w2t = tl.load(w2c + offs_all[:, None] * s2c0 + offs_j[None, :] * s2c1)
        accc += tl.dot(h1t, tl.trans(w2t))
    accc += tl.load(b2c + offs_all)[None, :]
    h2c = tl.where(accc > 0, accc, 0.0)
    w3ct = tl.load(w3c + offs_out[:, None] * s3c0 + offs_all[None, :] * s3c1,
                   mask=offs_out[:, None] < 1, other=0.0)
    accv = tl.dot(h2c, tl.trans(w3ct))
    accv += tl.load(b3c + offs_out, mask=offs_out < 1, other=0.0)[None, :]
    tl.store(yval_ptr + offs_m[:, None] * syv0 + offs_out[None, :] * syv1, accv,
             mask=(offs_m[:, None] < M) & (offs_out[None, :] < 1))


class Value_Net(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(Value_Net, self).__init__()
        self.fc1 = nn.Linear(observation_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)


class Policy_Net(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(Policy_Net, self).__init__()
        self.fc1 = nn.Linear(observation_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)


class DDPGNew(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(DDPGNew, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.actor = Policy_Net(self.observation_dim, self.action_dim)
        self.critic = Value_Net(self.observation_dim, self.action_dim)

    def forward(self, state):
        state = state.contiguous()
        M = state.shape[0]
        OBS = self.observation_dim
        ADIM = self.action_dim
        a = self.actor
        c = self.critic
        yact = torch.empty((M, ADIM), device=state.device, dtype=torch.float32)
        yval = torch.empty((M, 1), device=state.device, dtype=torch.float32)
        _ddpg_kernel[(1,)](
            state,
            a.fc1.weight, a.fc1.bias, a.fc2.weight, a.fc2.bias, a.fc3.weight, a.fc3.bias,
            c.fc1.weight, c.fc1.bias, c.fc2.weight, c.fc2.bias, c.fc3.weight, c.fc3.bias,
            yact, yval, M, OBS, ADIM,
            state.stride(0), state.stride(1),
            a.fc1.weight.stride(0), a.fc1.weight.stride(1),
            a.fc2.weight.stride(0), a.fc2.weight.stride(1),
            a.fc3.weight.stride(0), a.fc3.weight.stride(1),
            c.fc1.weight.stride(0), c.fc1.weight.stride(1),
            c.fc2.weight.stride(0), c.fc2.weight.stride(1),
            c.fc3.weight.stride(0), c.fc3.weight.stride(1),
            yact.stride(0), yact.stride(1), yval.stride(0), yval.stride(1),
            256, 16, 16, 64, 16, num_warps=4, num_stages=2)
        return yact, yval
