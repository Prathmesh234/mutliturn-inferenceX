import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(B_ptr, M_ptr, W_ptr, bias, O_ptr,
                  n0, n1, n2, n3,
                  N1: tl.constexpr, BLOCK_N: tl.constexpr):
    p = tl.program_id(0)          # over (i0, i2)
    n23 = n2 * n3
    n12 = n1 * n2
    n123 = n1 * n23
    i0 = p // n2
    i2 = p % n2

    offs = tl.arange(0, BLOCK_N)
    mask = offs < n3
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)

    acc = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)  # (i3, i4)
    for i1 in range(N1):
        base = i1 * n123 + i2 * n23
        # linear over h for each l (l == i3 axis)
        b2d = tl.load(B_ptr + base + offs[:, None] * n3 + offs[None, :],
                      mask=mask[:, None] & mask[None, :], other=0.0)
        L_vec = tl.sum(b2d * w[None, :], axis=1) + bias       # (n3,) over l

        row = i0 * n12 + i1 * n2 + i2
        ml = tl.load(M_ptr + row * n3 + offs, mask=mask, other=0.0)
        logit = tl.where(mask, ml + L_vec, -float("inf"))
        e = tl.exp(logit - tl.max(logit, axis=0))
        aw_vec = e / tl.sum(e, axis=0)                        # (n3,) over i3

        # B[i1,i2,i3,i4] block, reuse b2d layout (offs->i3, offs->i4)
        acc += aw_vec[:, None] * b2d

    out_off = p * (n3 * n3) + offs[:, None] * n3 + offs[None, :]
    tl.store(O_ptr + out_off, acc, mask=mask[:, None] & mask[None, :])


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
        nprog = n0 * n2
        BLOCK_N = triton.next_power_of_2(n3)
        _fused_kernel[(nprog,)](B, M, w, bias, out, n0, n1, n2, n3,
                                N1=n1, BLOCK_N=BLOCK_N, num_warps=1)
        return out
