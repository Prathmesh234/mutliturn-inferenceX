import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(QF, KF, WQ, WK, BQ, BK, S, PW, PB, OUT,
                  Lq, Lk, D, E, HD, Odim, scale,
                  BLOCK_HB: tl.constexpr, BLOCK_LK: tl.constexpr, BLOCK_D: tl.constexpr,
                  BLOCK_E: tl.constexpr, BLOCK_O: tl.constexpr, BLOCK_B: tl.constexpr,
                  HB: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0)
    offs_hb = tl.arange(0, BLOCK_HB)
    offs_j = tl.arange(0, BLOCK_LK)
    offs_d = tl.arange(0, BLOCK_D)
    offs_e = tl.arange(0, BLOCK_E)
    mask_hb = offs_hb < HB
    mask_j = offs_j < Lk
    mask_d = offs_d < D
    mask_e = offs_e < E

    bb = offs_hb % B
    hh = offs_hb // B
    wrow = hh[:, None] * D + offs_d[None, :]

    qflat = tl.load(QF + (bb[:, None] * Lq + i) * E + offs_e[None, :],
                    mask=mask_hb[:, None] & mask_e[None, :], other=0.0)
    wq = tl.load(WQ + wrow[:, :, None] * E + offs_e[None, None, :],
                 mask=mask_hb[:, None, None] & mask_d[None, :, None] & mask_e[None, None, :],
                 other=0.0)
    bq = tl.load(BQ + wrow, mask=mask_hb[:, None] & mask_d[None, :], other=0.0)
    qx = tl.sum(qflat[:, None, :] * wq, axis=2) + bq

    kflat = tl.load(KF + (bb[:, None, None] * Lk + offs_j[None, :, None]) * E + offs_e[None, None, :],
                    mask=mask_hb[:, None, None] & mask_j[None, :, None] & mask_e[None, None, :],
                    other=0.0)
    wk = tl.load(WK + wrow[:, :, None] * E + offs_e[None, None, :],
                 mask=mask_hb[:, None, None] & mask_d[None, :, None] & mask_e[None, None, :],
                 other=0.0)
    bk = tl.load(BK + wrow, mask=mask_hb[:, None] & mask_d[None, :], other=0.0)
    kfull = tl.sum(kflat[:, :, None, :] * wk[:, None, :, :], axis=3) + bk[:, None, :]

    score = tl.sum(qx[:, None, :] * kfull, axis=2) * scale
    score = tl.where(mask_hb[:, None] & mask_j[None, :], score, -float('inf'))
    m = tl.max(score, axis=0)
    e = tl.exp(score - m[None, :])
    ssum = tl.sum(e, axis=0)
    sm = e / ssum[None, :]
    sm = tl.where(mask_hb[:, None] & mask_j[None, :], sm, 0.0)
    tl.store(S + offs_hb[:, None] * (Lq * Lk) + i * Lk + offs_j[None, :], sm,
             mask=mask_hb[:, None] & mask_j[None, :])

    out = tl.sum(sm[:, :, None] * kfull, axis=1)

    offs_o = tl.arange(0, BLOCK_O)
    offs_b = tl.arange(0, BLOCK_B)
    mask_o = offs_o < Odim
    mask_b = offs_b < B
    pw = tl.load(PW + offs_o[None, :, None] * HD + hh[:, None, None] * D + offs_d[None, None, :],
                 mask=mask_hb[:, None, None] & mask_o[None, :, None] & mask_d[None, None, :],
                 other=0.0)
    partial = tl.sum(out[:, None, :] * pw, axis=2)
    sel = (offs_b[:, None] == bb[None, :]) & mask_b[:, None] & mask_hb[None, :]
    result = tl.sum(tl.where(sel[:, :, None], partial[None, :, :], 0.0), axis=1)
    pb = tl.load(PB + offs_o, mask=mask_o, other=0.0)
    result = result + pb[None, :]
    tl.store(OUT + (offs_b[:, None] * Lq + i) * Odim + offs_o[None, :],
             result, mask=mask_b[:, None] & mask_o[None, :])


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
        O = self.proj.weight.shape[0]
        HB = H * mb_size
        HD = H * D

        k_flat = k.reshape(-1, E).contiguous()
        q_flat = q.reshape(-1, E).contiguous()

        if self.score_function == 'dot_product':
            scale = 1.0
        elif self.score_function == 'scaled_dot_product':
            scale = 1.0 / math.sqrt(D)
        else:
            raise RuntimeError('score_function not supported by Triton impl')

        score = torch.empty((HB, q_len, k_len), device=q_flat.device, dtype=q_flat.dtype)
        output = torch.empty((mb_size * q_len, O), device=q_flat.device, dtype=q_flat.dtype)
        BLOCK_HB = triton.next_power_of_2(HB)
        BLOCK_LK = triton.next_power_of_2(k_len)
        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_E = triton.next_power_of_2(E)
        BLOCK_O = triton.next_power_of_2(O)
        BLOCK_B = triton.next_power_of_2(mb_size)
        _fused_kernel[(q_len,)](q_flat, k_flat, self.w_q.weight, self.w_k.weight,
                                self.w_q.bias, self.w_k.bias, score,
                                self.proj.weight, self.proj.bias, output,
                                q_len, k_len, D, E, HD, O, scale,
                                BLOCK_HB=BLOCK_HB, BLOCK_LK=BLOCK_LK, BLOCK_D=BLOCK_D,
                                BLOCK_E=BLOCK_E, BLOCK_O=BLOCK_O, BLOCK_B=BLOCK_B,
                                HB=HB, B=mb_size, num_warps=2)

        output = output.view(mb_size, q_len, O)
        output = self.dropout(output)
        return output, score
