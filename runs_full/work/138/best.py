import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _fused_kernel(src_ptr, trg_ptr, mask_ptr, out_ptr, B, M, K, F, tclip, scale,
                  inv_nce, inv_lgt,
                  BLOCK_B: tl.constexpr, BLOCK_F: tl.constexpr, BLOCK_K: tl.constexpr):
    offs_j = tl.arange(0, BLOCK_B)
    offs_f = tl.arange(0, BLOCK_F)
    offs_k = tl.arange(0, BLOCK_K)
    valid_j = offs_j < B
    valid_f = offs_f < F
    valid = valid_j[:, None] & valid_f[None, :]
    vk = offs_k < K
    v3 = vk[:, None, None] & valid_j[None, :, None] & valid_f[None, None, :]

    tot_nce = 0.0
    tot_sumsq = 0.0
    for i in range(0, B):
        src = tl.load(src_ptr + i * K + offs_k, mask=vk, other=0.0)
        toff = offs_k[:, None, None] * M + offs_j[None, :, None] * F + offs_f[None, None, :]
        trg = tl.load(trg_ptr + toff, mask=v3, other=0.0)
        S = tl.sum(src[:, None, None] * trg, axis=0) * scale
        tot_sumsq += tl.sum(tl.where(valid, S * S, 0.0))
        T = tclip * libdevice.tanh(S / tclip)
        mp_j = tl.load(mask_ptr + i * B + offs_j, mask=valid_j, other=0.0)
        mp = mp_j[:, None] + tl.zeros((BLOCK_B, BLOCK_F), tl.float32)
        mn = 1.0 - mp
        pos = tl.sum(tl.where(valid, mp * T, 0.0), axis=0)
        neg = mn * T - tclip * mp
        neg_max = tl.max(tl.where(valid, neg, float("-inf")))
        neg_sumexp = tl.sum(tl.where(valid, mn * tl.exp(neg - neg_max), 0.0))
        lse = tl.log(tl.exp(pos - neg_max) + neg_sumexp)
        nce = (pos - neg_max) - lse
        tot_nce += tl.sum(tl.where(valid_f, nce, 0.0))

    tl.store(out_ptr + 0, -tot_nce * inv_nce)
    tl.store(out_ptr + 1, tot_sumsq * inv_lgt)


class AmdimNCELossNew(nn.Module):
    def __init__(self, tclip):
        super().__init__()
        self.tclip = tclip

    def forward(self, anchor_representations, positive_representations, mask_mat):
        r_src = anchor_representations.contiguous().float()
        r_trg = positive_representations.contiguous().float()
        mask_mat = mask_mat.contiguous().float()
        B, K = r_src.shape
        M = r_trg.shape[1]
        F = M // B
        scale = 1.0 / (K ** 0.5)

        out = torch.empty(2, device=r_src.device, dtype=torch.float32)
        BLOCK_B = triton.next_power_of_2(B)
        BLOCK_F = triton.next_power_of_2(F)
        BLOCK_K = triton.next_power_of_2(K)
        inv_nce = 1.0 / (B * F)
        inv_lgt = 0.05 / (B * M)
        _fused_kernel[(1,)](r_src, r_trg, mask_mat, out, B, M, K, F,
                            float(self.tclip), scale, inv_nce, inv_lgt,
                            BLOCK_B=BLOCK_B, BLOCK_F=BLOCK_F, BLOCK_K=BLOCK_K, num_warps=1)
        return out[0], out[1]
