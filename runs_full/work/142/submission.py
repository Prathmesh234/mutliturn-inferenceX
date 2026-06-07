import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused(s_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr,
           w3_ptr, b3_ptr, w4_ptr, b4_ptr,
           sc1_ptr, sc2_ptr, out_ptr,
           M, K0, A,
           H: tl.constexpr, BM: tl.constexpr, BK0: tl.constexpr,
           BK: tl.constexpr, BA: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    mmask = offs_m < M
    offs_h = tl.arange(0, H)

    offs_k0 = tl.arange(0, BK0)
    s = tl.load(s_ptr + offs_m[:, None] * K0 + offs_k0[None, :],
                mask=mmask[:, None] & (offs_k0[None, :] < K0), other=0.0)
    w1 = tl.load(w1_ptr + offs_h[:, None] * K0 + offs_k0[None, :],
                 mask=offs_k0[None, :] < K0, other=0.0)
    b1 = tl.load(b1_ptr + offs_h)
    h1 = tl.maximum(tl.dot(s, tl.trans(w1)) + b1[None, :], 0.0)
    tl.store(sc1_ptr + offs_m[:, None] * H + offs_h[None, :], h1, mask=mmask[:, None])
    tl.debug_barrier()

    acc2 = tl.zeros((BM, H), dtype=tl.float32)
    for k in range(0, H, BK):
        ok = k + tl.arange(0, BK)
        a = tl.load(sc1_ptr + offs_m[:, None] * H + ok[None, :], mask=mmask[:, None], other=0.0)
        w2 = tl.load(w2_ptr + offs_h[:, None] * H + ok[None, :])
        acc2 += tl.dot(a, tl.trans(w2))
    b2 = tl.load(b2_ptr + offs_h)
    h2 = tl.maximum(acc2 + b2[None, :], 0.0)
    tl.store(sc2_ptr + offs_m[:, None] * H + offs_h[None, :], h2, mask=mmask[:, None])
    tl.debug_barrier()

    acc3 = tl.zeros((BM, H), dtype=tl.float32)
    for k in range(0, H, BK):
        ok = k + tl.arange(0, BK)
        a = tl.load(sc2_ptr + offs_m[:, None] * H + ok[None, :], mask=mmask[:, None], other=0.0)
        w3 = tl.load(w3_ptr + offs_h[:, None] * H + ok[None, :])
        acc3 += tl.dot(a, tl.trans(w3))
    b3 = tl.load(b3_ptr + offs_h)
    h3 = tl.maximum(acc3 + b3[None, :], 0.0)
    tl.store(sc1_ptr + offs_m[:, None] * H + offs_h[None, :], h3, mask=mmask[:, None])
    tl.debug_barrier()

    offs_a = tl.arange(0, BA)
    acc4 = tl.zeros((BM, BA), dtype=tl.float32)
    for k in range(0, H, BK):
        ok = k + tl.arange(0, BK)
        a = tl.load(sc1_ptr + offs_m[:, None] * H + ok[None, :], mask=mmask[:, None], other=0.0)
        w4 = tl.load(w4_ptr + offs_a[:, None] * H + ok[None, :], mask=offs_a[:, None] < A, other=0.0)
        acc4 += tl.dot(a, tl.trans(w4))
    b4 = tl.load(b4_ptr + offs_a, mask=offs_a < A, other=0.0)
    out = acc4 + b4[None, :]
    tl.store(out_ptr + offs_m[:, None] * A + offs_a[None, :],
             out, mask=mmask[:, None] & (offs_a[None, :] < A))


class QNetworkNew(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetworkNew, self).__init__()
        self.seed = torch.manual_seed(seed)
        hidden_units = 512
        self.fc1 = nn.Linear(state_size, hidden_units)
        self.do1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.do2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.do3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(hidden_units, action_size)

    def forward(self, state):
        orig_shape = state.shape
        K0 = orig_shape[-1]
        x = state.reshape(-1, K0).contiguous()
        M = x.shape[0]
        H = 512
        A = self.fc4.weight.shape[0]
        BA = max(16, triton.next_power_of_2(A))
        BK0 = max(16, triton.next_power_of_2(K0))
        sc1 = torch.empty((M, H), device=x.device, dtype=torch.float32)
        sc2 = torch.empty((M, H), device=x.device, dtype=torch.float32)
        out = torch.empty((M, A), device=x.device, dtype=x.dtype)
        BM = 16
        grid = (triton.cdiv(M, BM),)
        _fused[grid](x, self.fc1.weight, self.fc1.bias,
                     self.fc2.weight, self.fc2.bias,
                     self.fc3.weight, self.fc3.bias,
                     self.fc4.weight, self.fc4.bias,
                     sc1, sc2, out, M, K0, A,
                     H=H, BM=BM, BK0=BK0, BK=64, BA=BA,
                     num_warps=2, num_stages=2)
        return out.reshape(*orig_shape[:-1], A)
