import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(AW_unused, B_ptr, M_ptr, W_ptr, bias, O_ptr,
                  n0, n1, n2, n3,
                  N1: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr):
    p = tl.program_id(0)
    n23 = n2 * n3
    n12 = n1 * n2
    n123 = n1 * n23
    i0 = p // n23
    rem = p % n23
    i2 = rem // n3
    i3 = rem % n3

    offs_l = tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, BLOCK_H)
    mask_l = offs_l < n3
    mask_h = offs_h < n3        # H == n3
    w = tl.load(W_ptr + offs_h, mask=mask_h, other=0.0)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i1 in range(N1):
        base_l = i1 * n123 + i2 * n23
        b2d = tl.load(B_ptr + base_l + offs_l[:, None] * n3 + offs_h[None, :],
                      mask=mask_l[:, None] & mask_h[None, :], other=0.0)
        L_vec = tl.sum(b2d * w[None, :], axis=1) + bias

        row = i0 * n12 + i1 * n2 + i2
        ml = tl.load(M_ptr + row * n3 + offs_l, mask=mask_l, other=0.0)
        logit = tl.where(mask_l, ml + L_vec, -float("inf"))
        mx = tl.max(logit, axis=0)
        e = tl.exp(logit - mx)
        aw_vec = e / tl.sum(e, axis=0)
        aw_s = tl.sum(tl.where(offs_l == i3, aw_vec, 0.0), axis=0)

        bv = tl.load(B_ptr + i1 * n123 + i2 * n23 + i3 * n3 + offs_l,
                     mask=mask_l, other=0.0)
        acc += aw_s * bv

    tl.store(O_ptr + p * n3 + offs_l, acc, mask=mask_l)


class SelfAttentionPoolingNew(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionPoolingNew, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask):
        B = batch_rep.contiguous()
        M = att_mask.contiguous()
        n0, n1, n2, n3 = B.shape
        w = self.W.weight.contiguous().view(-1)
        bias = float(self.W.bias.item())

        out = torch.empty(n0, n2, n3, n3, device=B.device, dtype=torch.float32)
        nprog = n0 * n2 * n3
        BLOCK_N = triton.next_power_of_2(n3)
        BLOCK_H = triton.next_power_of_2(n3)
        _fused_kernel[(nprog,)](0, B, M, w, bias, out, n0, n1, n2, n3,
                                N1=n1, BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H,
                                num_warps=1)
        return out
