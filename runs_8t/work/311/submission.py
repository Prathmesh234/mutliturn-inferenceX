import math
import torch
import numpy as np
from torch import nn
import triton
import triton.language as tl


def _next_pow2(x):
    return 1 if x <= 1 else (1 << (x - 1).bit_length())


@triton.jit
def _fused_kernel(QSRC, KVSRC, WQ, WK, WV, WOUT, OUT, MASK,
                  B, Nq, T,
                  DI: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                  DE: tl.constexpr, H: tl.constexpr, norm,
                  BLOCK_Q: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_DI: tl.constexpr,
                  BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
                  BLOCK_DE: tl.constexpr, HAS_MASK: tl.constexpr):
    b = tl.program_id(0)

    offs_q = tl.arange(0, BLOCK_Q)
    offs_t = tl.arange(0, BLOCK_T)
    offs_di = tl.arange(0, BLOCK_DI)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    offs_de = tl.arange(0, BLOCK_DE)
    mask_q = offs_q < Nq
    mask_t = offs_t < T
    mask_di = offs_di < DI
    mask_k = offs_k < DK
    mask_v = offs_v < DV
    mask_de = offs_de < DE

    # q rows (BLOCK_Q, BLOCK_DI)
    xq = tl.load(QSRC + (b * Nq + offs_q)[:, None] * DI + offs_di[None, :],
                 mask=mask_q[:, None] & mask_di[None, :], other=0.0)
    # kv rows (BLOCK_T, BLOCK_DI)
    xkv = tl.load(KVSRC + (b * T + offs_t)[:, None] * DI + offs_di[None, :],
                  mask=mask_t[:, None] & mask_di[None, :], other=0.0)

    if HAS_MASK:
        m_qt = tl.load(MASK + (b * Nq + offs_q)[:, None] * T + offs_t[None, :],
                       mask=mask_q[:, None] & mask_t[None, :], other=0).to(tl.int1)

    out_acc = tl.zeros((BLOCK_Q, BLOCK_DE), tl.float32)

    for h in tl.static_range(H):
        wq = tl.load(WQ + h * DI * DK + offs_di[:, None] * DK + offs_k[None, :],
                     mask=mask_di[:, None] & mask_k[None, :], other=0.0)
        q_h = tl.sum(xq[:, :, None] * wq[None, :, :], axis=1)  # (BLOCK_Q, BLOCK_K)

        wk = tl.load(WK + h * DI * DK + offs_di[:, None] * DK + offs_k[None, :],
                     mask=mask_di[:, None] & mask_k[None, :], other=0.0)
        k_h = tl.sum(xkv[:, :, None] * wk[None, :, :], axis=1)  # (BLOCK_T, BLOCK_K)

        # compat (BLOCK_Q, BLOCK_T)
        compat = tl.sum(q_h[:, None, :] * k_h[None, :, :], axis=2) * norm
        compat = tl.where(mask_t[None, :], compat, float("-inf"))
        if HAS_MASK:
            compat = tl.where(m_qt, float("-inf"), compat)
        mx = tl.max(compat, axis=1)  # (BLOCK_Q,)
        p = tl.exp(compat - mx[:, None])
        if HAS_MASK:
            p = tl.where(mask_t[None, :], p, 0.0)
        s = tl.sum(p, axis=1)  # (BLOCK_Q,)
        attn = p / s[:, None]  # (BLOCK_Q, BLOCK_T)

        wv = tl.load(WV + h * DI * DV + offs_di[:, None] * DV + offs_v[None, :],
                     mask=mask_di[:, None] & mask_v[None, :], other=0.0)
        v_h = tl.sum(xkv[:, :, None] * wv[None, :, :], axis=1)  # (BLOCK_T, BLOCK_V)

        head_h = tl.sum(attn[:, :, None] * v_h[None, :, :], axis=1)  # (BLOCK_Q, BLOCK_V)

        wout = tl.load(WOUT + h * DV * DE + offs_v[:, None] * DE + offs_de[None, :],
                       mask=mask_v[:, None] & mask_de[None, :], other=0.0)
        out_acc += tl.sum(head_h[:, :, None] * wout[None, :, :], axis=1)  # (BLOCK_Q, BLOCK_DE)

    tl.store(OUT + (b * Nq + offs_q)[:, None] * DE + offs_de[None, :],
             out_acc, mask=mask_q[:, None] & mask_de[None, :])


class MultiHeadAttentionNew(nn.Module):

    def __init__(self, n_heads, input_dim, embed_dim, val_dim=None, key_dim=None):
        super(MultiHeadAttentionNew, self).__init__()
        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.norm_factor = 1 / math.sqrt(key_dim)
        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))
        self.init_parameters()
        self._B_DI = _next_pow2(input_dim)
        self._B_K = _next_pow2(key_dim)
        self._B_V = _next_pow2(val_dim)
        self._B_DE = _next_pow2(embed_dim)

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, queries, data=None, mask=None):
        if data is None:
            data = queries
        batch_size, task_size, input_dim = data.size()
        n_query = queries.size(1)

        q_src = queries.contiguous().view(-1, input_dim)
        kv_src = data.contiguous().view(-1, input_dim)

        Out = torch.empty((batch_size * n_query, self.embed_dim),
                          device=queries.device, dtype=queries.dtype)

        has_mask = mask is not None
        if has_mask:
            mask_t = mask.contiguous().view(batch_size, n_query, task_size).to(torch.int8)
        else:
            mask_t = q_src

        _fused_kernel[(batch_size,)](
            q_src, kv_src, self.W_query, self.W_key, self.W_val, self.W_out, Out, mask_t,
            batch_size, n_query, task_size,
            input_dim, self.key_dim, self.val_dim, self.embed_dim, self.n_heads,
            self.norm_factor,
            BLOCK_Q=_next_pow2(n_query), BLOCK_T=_next_pow2(task_size), BLOCK_DI=self._B_DI,
            BLOCK_K=self._B_K, BLOCK_V=self._B_V, BLOCK_DE=self._B_DE,
            HAS_MASK=has_mask, num_warps=4)

        return Out.view(batch_size, n_query, self.embed_dim)
