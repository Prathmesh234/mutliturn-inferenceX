import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _ac_kernel(x_ptr, w1_ptr, b1_ptr, wa_ptr, ba_ptr, wc_ptr, bc_ptr,
               a_ptr, c_ptr, M,
               IN_FEAT: tl.constexpr, HIDDEN: tl.constexpr, N_ACTIONS: tl.constexpr,
               BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_A: tl.constexpr):
    row = tl.program_id(0)
    if row >= M:
        return

    offs_k = tl.arange(0, BLOCK_K)
    offs_h = tl.arange(0, BLOCK_H)
    offs_a = tl.arange(0, BLOCK_A)
    mk = offs_k < IN_FEAT
    mh = offs_h < HIDDEN
    ma = offs_a < N_ACTIONS

    # fc1 + relu
    xrow = tl.load(x_ptr + row * IN_FEAT + offs_k, mask=mk, other=0.0)
    w1 = tl.load(w1_ptr + offs_h[:, None] * IN_FEAT + offs_k[None, :],
                 mask=mh[:, None] & mk[None, :], other=0.0)
    h = tl.sum(w1 * xrow[None, :], axis=1) + tl.load(b1_ptr + offs_h, mask=mh, other=0.0)
    h = tl.maximum(h, 0.0)
    h = tl.where(mh, h, 0.0)

    # actor head + log_softmax
    wa = tl.load(wa_ptr + offs_a[:, None] * HIDDEN + offs_h[None, :],
                 mask=ma[:, None] & mh[None, :], other=0.0)
    la = tl.sum(wa * h[None, :], axis=1) + tl.load(ba_ptr + offs_a, mask=ma, other=0.0)
    la_m = tl.where(ma, la, -float('inf'))
    m = tl.max(la_m, axis=0)
    s = tl.sum(tl.exp(la_m - m), axis=0)
    lse = m + tl.log(s)
    tl.store(a_ptr + row * N_ACTIONS + offs_a, la - lse, mask=ma)

    # critic head
    wc = tl.load(wc_ptr + offs_h, mask=mh, other=0.0)
    c = tl.sum(wc * h, axis=0) + tl.load(bc_ptr)
    tl.store(c_ptr + row, c)


class ActorCriticMLPNew(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_shape[0], hidden_size)
        self.actor_head = nn.Linear(hidden_size, n_actions)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.float()
        in_feat = self.fc1.in_features
        hidden = self.fc1.out_features
        n_actions = self.actor_head.out_features
        batch_shape = x.shape[:-1]
        xf = x.reshape(-1, in_feat).contiguous()
        M = xf.shape[0]

        a = torch.empty((M, n_actions), device=x.device, dtype=torch.float32)
        c = torch.empty((M, 1), device=x.device, dtype=torch.float32)

        def npow2(n):
            return 1 << (max(n - 1, 0)).bit_length()

        grid = (M,)
        _ac_kernel[grid](
            xf, self.fc1.weight, self.fc1.bias,
            self.actor_head.weight, self.actor_head.bias,
            self.critic_head.weight, self.critic_head.bias,
            a, c, M,
            IN_FEAT=in_feat, HIDDEN=hidden, N_ACTIONS=n_actions,
            BLOCK_K=npow2(in_feat), BLOCK_H=npow2(hidden), BLOCK_A=npow2(n_actions),
            num_warps=2,
        )
        return a.reshape(*batch_shape, n_actions), c.reshape(*batch_shape, 1)
