import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_kernel(B_ptr, W_ptr, bias, L_ptr, H, BLOCK_H: tl.constexpr):
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    b = tl.load(B_ptr + m * H + offs, mask=mask, other=0.0)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    acc = tl.sum(b * w) + bias
    tl.store(L_ptr + m, acc)


@triton.jit
def _softmax_kernel(M_ptr, L_ptr, AW_ptr, n1, n2, n3, BLOCK_N: tl.constexpr):
    r = tl.program_id(0)
    n12 = n1 * n2
    rem = r % n12
    j = rem // n2
    k = rem % n2
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n3
    ml = tl.load(M_ptr + r * n3 + offs, mask=mask, other=0.0)
    ll = tl.load(L_ptr + j * n12 + k * n2 + offs, mask=mask, other=0.0)
    logit = ml + ll
    logit = tl.where(mask, logit, -float("inf"))
    mx = tl.max(logit, axis=0)
    e = tl.exp(logit - mx)
    s = tl.sum(e, axis=0)
    aw = e / s
    tl.store(AW_ptr + r * n3 + offs, aw, mask=mask)


@triton.jit
def _weighted_kernel(AW_ptr, B_ptr, O_ptr, n1, n2, n3,
                     N1: tl.constexpr, BLOCK_N: tl.constexpr):
    p = tl.program_id(0)
    n23 = n2 * n3
    i0 = p // n23
    rem = p % n23
    i2 = rem // n3
    i3 = rem % n3
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n3
    n123 = n1 * n2 * n3
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i1 in range(N1):
        aw = tl.load(AW_ptr + i0 * n123 + i1 * n23 + i2 * n3 + i3)
        bv = tl.load(B_ptr + i1 * n123 + i2 * n23 + i3 * n3 + offs,
                     mask=mask, other=0.0)
        acc += aw * bv
    tl.store(O_ptr + p * n3 + offs, acc, mask=mask)


class SelfAttentionPoolingNew(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionPoolingNew, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask):
        B = batch_rep.contiguous()
        M = att_mask.contiguous()
        n0, n1, n2, n3 = B.shape
        H = n3

        w = self.W.weight.contiguous().view(-1)   # (H,)
        bias = float(self.W.bias.item())

        rows = n0 * n1 * n2
        L = torch.empty(rows, device=B.device, dtype=torch.float32)
        BLOCK_H = triton.next_power_of_2(H)
        _linear_kernel[(rows,)](B, w, bias, L, H, BLOCK_H=BLOCK_H, num_warps=1)

        BLOCK_N = triton.next_power_of_2(n3)
        AW = torch.empty(n0 * n1 * n2 * n3, device=B.device, dtype=torch.float32)
        _softmax_kernel[(rows,)](M, L, AW, n1, n2, n3, BLOCK_N=BLOCK_N, num_warps=1)

        out = torch.empty(n0, n2, n3, n3, device=B.device, dtype=torch.float32)
        nprog = n0 * n2 * n3
        _weighted_kernel[(nprog,)](AW, B, out, n1, n2, n3,
                                   N1=n1, BLOCK_N=BLOCK_N, num_warps=1)
        return out
