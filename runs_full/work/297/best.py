import math
import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(ref_ptr, Wr_ptr, br_ptr,
                  query_ptr, Wq_ptr, bq_ptr,
                  v_ptr, e_ptr, logits_ptr,
                  B, D, K, L, C, USE_TANH: tl.constexpr,
                  BD: tl.constexpr, BK: tl.constexpr, BL: tl.constexpr):
    b = tl.program_id(0)
    d = tl.arange(0, BD)
    k = tl.arange(0, BK)
    l = tl.arange(0, BL)
    md = d < D
    mk = k < K
    ml = l < L

    Wr = tl.load(Wr_ptr + d[:, None] * K + k[None, :], mask=md[:, None] & mk[None, :], other=0.0)
    refp = tl.load(ref_ptr + l[None, :] * (B * K) + b * K + k[:, None],
                   mask=mk[:, None] & ml[None, :], other=0.0)
    acc = tl.sum(Wr[:, :, None] * refp[None, :, :], axis=1)  # [BD, BL]
    br = tl.load(br_ptr + d, mask=md, other=0.0)
    e = acc + br[:, None]
    tl.store(e_ptr + b * (D * L) + d[:, None] * L + l[None, :], e,
             mask=md[:, None] & ml[None, :])

    Wq = tl.load(Wq_ptr + d[:, None] * K + k[None, :], mask=md[:, None] & mk[None, :], other=0.0)
    qx = tl.load(query_ptr + b * K + k, mask=mk, other=0.0)
    q = tl.sum(Wq * qx[None, :], axis=1) + tl.load(bq_ptr + d, mask=md, other=0.0)

    t = 2.0 * tl.sigmoid(2.0 * (q[:, None] + e)) - 1.0  # [BD, BL]
    v = tl.load(v_ptr + d, mask=md, other=0.0)
    u = tl.sum(v[:, None] * t, axis=0)  # [BL]
    if USE_TANH:
        u = C * (2.0 * tl.sigmoid(2.0 * u) - 1.0)
    tl.store(logits_ptr + b * L + l, u, mask=ml)


def _next_pow2(x):
    return 1 << (max(x, 1) - 1).bit_length()


class AttentionNew(nn.Module):
    """A generic attention module for a decoder in seq2seq"""

    def __init__(self, dim, use_tanh=False, C=10):
        super(AttentionNew, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C
        self.tanh = nn.Tanh()
        self.v = nn.Parameter(torch.FloatTensor(dim))
        self.v.data.uniform_(-(1.0 / math.sqrt(dim)), 1.0 / math.sqrt(dim))

    def forward(self, query, ref):
        L, B, K = ref.shape
        D = self.project_query.out_features
        ref = ref if ref.is_contiguous() else ref.contiguous()
        query = query if query.is_contiguous() else query.contiguous()

        e = torch.empty((B, D, L), device=query.device, dtype=torch.float32)
        logits = torch.empty((B, L), device=query.device, dtype=torch.float32)

        BD = _next_pow2(D)
        BK = _next_pow2(K)
        BL = _next_pow2(L)

        Wr = self.project_ref.weight.view(D, K)
        br = self.project_ref.bias
        Wq = self.project_query.weight
        bq = self.project_query.bias

        grid = (B,)
        _fused_kernel[grid](ref, Wr, br, query, Wq, bq, self.v, e, logits,
                            B, D, K, L, float(self.C), self.use_tanh,
                            BD=BD, BK=BK, BL=BL, num_warps=1)
        return e, logits
