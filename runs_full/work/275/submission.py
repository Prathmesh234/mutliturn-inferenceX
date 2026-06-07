import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mask_head_kernel(feat_ptr, w_ptr, b_ptr, out_ptr,
                      B, N, EMBED: tl.constexpr,
                      BLOCK_B: tl.constexpr, BLOCK_N: tl.constexpr):
    offs_b = tl.arange(0, BLOCK_B)
    offs_n = tl.arange(0, BLOCK_N)
    mask_b = offs_b < B
    mask_n = offs_n < N
    mask = mask_b[:, None] & mask_n[None, :]
    bias = tl.load(b_ptr)
    acc = tl.zeros((BLOCK_B, BLOCK_N), tl.float32) + bias
    rowbase = offs_b[:, None] * (N * EMBED) + offs_n[None, :] * EMBED
    for d in tl.static_range(EMBED):
        f = tl.load(feat_ptr + rowbase + d, mask=mask, other=0.0)
        w = tl.load(w_ptr + d)
        acc += f * w
    acc = tl.where(mask, acc, -float('inf'))
    m = tl.max(acc, axis=1)
    e = tl.exp(acc - m[:, None])
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=1)
    out = e / s[:, None]
    tl.store(out_ptr + offs_b[:, None] * N + offs_n[None, :], out, mask=mask)


class RobertaMaskLeanerHeadNew(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.dense = nn.Linear(embed_dim, 1)
        self.embed_dim = embed_dim

    def forward(self, features, **kwargs):
        embed = features.size(-1)
        batch = features.size(0)
        feat = features.contiguous().view(batch, -1, embed)
        N = feat.size(1)
        out = torch.empty((batch, N), device=features.device, dtype=features.dtype)
        w = self.dense.weight.view(-1).contiguous()
        b = self.dense.bias
        BLOCK_B = triton.next_power_of_2(batch)
        BLOCK_N = triton.next_power_of_2(N)
        _mask_head_kernel[(1,)](feat, w, b, out, batch, N, embed,
                                BLOCK_B=BLOCK_B, BLOCK_N=BLOCK_N, num_warps=1)
        return out
