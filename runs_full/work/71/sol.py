import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_kernel(X, W, Bz, Y, R, K, N, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    mask_k = offs_k < K
    mask_n = offs_n < N
    x = tl.load(X + pid * K + offs_k, mask=mask_k, other=0.0)
    w = tl.load(W + offs_n[:, None] * K + offs_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :], other=0.0)
    acc = tl.sum(x[None, :] * w, axis=1)
    b = tl.load(Bz + offs_n, mask=mask_n, other=0.0)
    acc = acc + b
    tl.store(Y + pid * N + offs_n, acc, mask=mask_n)


@triton.jit
def _score_kernel(Q, Kk, S, Lq, Lk, D, scale,
                  BLOCK_LK: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    offs_j = tl.arange(0, BLOCK_LK)
    mask_d = offs_d < D
    mask_j = offs_j < Lk
    qv = tl.load(Q + pid_b * (Lq * D) + pid_i * D + offs_d, mask=mask_d, other=0.0)
    kv = tl.load(Kk + pid_b * (Lk * D) + offs_j[:, None] * D + offs_d[None, :],
                 mask=mask_j[:, None] & mask_d[None, :], other=0.0)
    acc = tl.sum(qv[None, :] * kv, axis=1) * scale
    tl.store(S + pid_b * (Lq * Lk) + pid_i * Lk + offs_j, acc, mask=mask_j)


@triton.jit
def _softmax_dim0_kernel(S, HB, P, BLOCK_HB: tl.constexpr):
    p = tl.program_id(0)
    offs_b = tl.arange(0, BLOCK_HB)
    mask_b = offs_b < HB
    ptrs = S + offs_b * P + p
    x = tl.load(ptrs, mask=mask_b, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    tl.store(ptrs, e / s, mask=mask_b)


@triton.jit
def _output_kernel(S, Kk, O, Lq, Lk, D,
                   BLOCK_LK: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    offs_j = tl.arange(0, BLOCK_LK)
    mask_d = offs_d < D
    mask_j = offs_j < Lk
    sv = tl.load(S + pid_b * (Lq * Lk) + pid_i * Lk + offs_j, mask=mask_j, other=0.0)
    kv = tl.load(Kk + pid_b * (Lk * D) + offs_j[:, None] * D + offs_d[None, :],
                 mask=mask_j[:, None] & mask_d[None, :], other=0.0)
    acc = tl.sum(sv[:, None] * kv, axis=0)
    tl.store(O + pid_b * (Lq * D) + pid_i * D + offs_d, acc, mask=mask_d)


def _linear(x, weight, bias):
    R, K = x.shape
    N = weight.shape[0]
    y = torch.empty((R, N), device=x.device, dtype=x.dtype)
    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_N = triton.next_power_of_2(N)
    _linear_kernel[(R,)](x, weight, bias, y, R, K, N,
                         BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N, num_warps=4)
    return y


class AttentionNew(nn.Module):

    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1,
                 score_function='dot_product', dropout=0):
        super(AttentionNew, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]
        k_len = k.shape[1]
        q_len = q.shape[1]
        H = self.n_head
        D = self.hidden_dim
        E = self.embed_dim
        HB = H * mb_size

        k_flat = k.reshape(-1, E).contiguous()
        q_flat = q.reshape(-1, E).contiguous()

        kx_flat = _linear(k_flat, self.w_k.weight, self.w_k.bias)
        qx_flat = _linear(q_flat, self.w_q.weight, self.w_q.bias)

        # [B, L, H, D] -> [H, B, L, D] -> [HB, L, D]
        kx = kx_flat.view(mb_size, k_len, H, D).permute(2, 0, 1, 3).contiguous().view(HB, k_len, D)
        qx = qx_flat.view(mb_size, q_len, H, D).permute(2, 0, 1, 3).contiguous().view(HB, q_len, D)

        if self.score_function == 'dot_product':
            scale = 1.0
        elif self.score_function == 'scaled_dot_product':
            scale = 1.0 / math.sqrt(D)
        else:
            raise RuntimeError('score_function not supported by Triton impl')

        score = torch.empty((HB, q_len, k_len), device=kx.device, dtype=kx.dtype)
        BLOCK_LK = triton.next_power_of_2(k_len)
        BLOCK_D = triton.next_power_of_2(D)
        _score_kernel[(HB, q_len)](qx, kx, score, q_len, k_len, D, scale,
                                   BLOCK_LK=BLOCK_LK, BLOCK_D=BLOCK_D, num_warps=4)

        # softmax over dim 0
        P = q_len * k_len
        BLOCK_HB = triton.next_power_of_2(HB)
        _softmax_dim0_kernel[(P,)](score, HB, P, BLOCK_HB=BLOCK_HB, num_warps=4)

        out = torch.empty((HB, q_len, D), device=kx.device, dtype=kx.dtype)
        _output_kernel[(HB, q_len)](score, kx, out, q_len, k_len, D,
                                    BLOCK_LK=BLOCK_LK, BLOCK_D=BLOCK_D, num_warps=4)

        # [HB, Lq, D] = [H, B, Lq, D] -> [B, Lq, H, D] -> [B, Lq, H*D]
        output = out.view(H, mb_size, q_len, D).permute(1, 2, 0, 3).contiguous().view(mb_size * q_len, H * D)
        output = _linear(output, self.proj.weight, self.proj.bias)
        output = output.view(mb_size, q_len, -1)
        output = self.dropout(output)
        return output, score
