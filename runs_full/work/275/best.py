import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mask_head_kernel(feat_ptr, w_ptr, b_ptr, out_ptr,
                      N, EMBED: tl.constexpr,
                      BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < N
    mask_d = offs_d < EMBED
    base = pid * N * EMBED
    ptrs = base + offs_n[:, None] * EMBED + offs_d[None, :]
    feat = tl.load(feat_ptr + ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
    w = tl.load(w_ptr + offs_d, mask=mask_d, other=0.0)
    bias = tl.load(b_ptr)
    acc = tl.sum(feat * w[None, :], axis=1) + bias
    acc = tl.where(mask_n, acc, -float('inf'))
    m = tl.max(acc, axis=0)
    e = tl.exp(acc - m)
    e = tl.where(mask_n, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(out_ptr + pid * N + offs_n, out, mask=mask_n)


class RobertaMaskLeanerHeadNew(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.dense = nn.Linear(embed_dim, 1)
        self.embed_dim = embed_dim

    def forward(self, features, **kwargs):
        embed = features.size(-1)
        batch = features.size(0)
        feat = features.contiguous().view(-1, embed)
        N = feat.size(0) // batch
        feat = feat.view(batch, N, embed)
        out = torch.empty((batch, N), device=features.device, dtype=features.dtype)
        w = self.dense.weight.view(-1).contiguous()
        b = self.dense.bias
        BLOCK_N = triton.next_power_of_2(N)
        BLOCK_D = triton.next_power_of_2(embed)
        _mask_head_kernel[(batch,)](feat, w, b, out, N, embed,
                                    BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, num_warps=1)
        return out
