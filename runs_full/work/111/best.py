import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(qs_ptr, hs_ptr, W_ptr, B_ptr, O_ptr,
                  Q, S, QSEQ, NSEQ, H, F,
                  BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)
    x = pid % QSEQ
    tmp = pid // QSEQ
    j = tmp % S
    i = tmp // S
    offs_h = tl.arange(0, BLOCK_H)
    offs_n = tl.arange(0, BLOCK_N)
    offs_f = tl.arange(0, BLOCK_F)
    mask_h = offs_h < H
    mask_n = offs_n < NSEQ
    mask_f = offs_f < F

    Wt = tl.load(W_ptr + offs_h[:, None] * F + offs_f[None, :],
                 mask=mask_h[:, None] & mask_f[None, :], other=0.0)  # [BH,BF]
    b = tl.load(B_ptr + offs_h, mask=mask_h, other=0.0)              # [BH]

    qx = tl.load(qs_ptr + (i * QSEQ + x) * F + offs_f, mask=mask_f, other=0.0)  # [BF]
    qrow = tl.sum(qx[None, :] * Wt, axis=1) + b                                 # [BH]

    h_rows = j * NSEQ + offs_n
    hstile = tl.load(hs_ptr + h_rows[:, None] * F + offs_f[None, :],
                     mask=mask_n[:, None] & mask_f[None, :], other=0.0)  # [BN,BF]
    # hW[y,h] = sum_f hstile[y,f]*Wt[h,f] + b[h]
    hW = tl.sum(hstile[:, None, :] * Wt[None, :, :], axis=2) + b[None, :]  # [BN,BH]

    z = qrow[None, :] * hW
    e2 = tl.exp(-2.0 * z)
    prod = (1.0 - e2) / (1.0 + e2)
    g = tl.sum(prod, axis=1)              # [BN]
    g = tl.where(mask_n, g, -float('inf'))
    g_max = tl.max(g, axis=0)
    e = tl.exp(g - g_max)
    e = tl.where(mask_n, e, 0.0)
    att = e / tl.sum(e, axis=0)           # [BN]

    out = tl.sum(att[:, None] * hstile, axis=0)  # [BF]
    o_off = ((i * S + j) * QSEQ + x) * F + offs_f
    tl.store(O_ptr + o_off, out, mask=mask_f)


class ItemQueryAttentionNew(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(ItemQueryAttentionNew, self).__init__()
        self.W = nn.Linear(feature_size, hidden_size)

    def forward(self, qs, hs):
        Q, QSEQ, F = qs.shape
        S, NSEQ, _ = hs.shape
        H = self.W.out_features
        qs_c = qs.contiguous()
        hs_c = hs.contiguous()
        Wt = self.W.weight.contiguous()
        Bt = self.W.bias.contiguous()
        BLOCK_F = triton.next_power_of_2(F)
        BLOCK_H = triton.next_power_of_2(H)
        BLOCK_N = triton.next_power_of_2(NSEQ)
        out = torch.empty((Q, S, QSEQ, F), device=qs.device, dtype=qs.dtype)
        _fused_kernel[(Q * S * QSEQ,)](qs_c, hs_c, Wt, Bt, out, Q, S, QSEQ, NSEQ, H, F,
                                       BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H, BLOCK_F=BLOCK_F,
                                       num_warps=1)
        return out
