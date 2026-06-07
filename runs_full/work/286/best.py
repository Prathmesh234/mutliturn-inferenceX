import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x1_ptr, x2_ptr, out_ptr, B, F,
                  BLOCK_B: tl.constexpr, BLOCK_F: tl.constexpr):
    offs_b = tl.arange(0, BLOCK_B)
    offs_f = tl.arange(0, BLOCK_F)
    mb = offs_b < B
    mf = offs_f < F
    ptrs = offs_b[:, None] * F + offs_f[None, :]
    m2d = mb[:, None] & mf[None, :]

    x1 = tl.load(x1_ptr + ptrs, mask=m2d, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + ptrs, mask=m2d, other=0.0).to(tl.float32)

    mean1 = tl.sum(x1, axis=0) / B
    mean2 = tl.sum(x2, axis=0) / B
    c1 = tl.where(m2d, x1 - mean1[None, :], 0.0)
    c2 = tl.where(m2d, x2 - mean2[None, :], 0.0)

    nrm1 = tl.sqrt(tl.sum(c1 * c1, axis=1)) + 1e-06
    nrm2 = tl.sqrt(tl.sum(c2 * c2, axis=1)) + 1e-06
    n1 = c1 / nrm1[:, None]
    n2 = c2 / nrm2[:, None]

    acc = tl.zeros((BLOCK_F, BLOCK_F), tl.float32)
    for b in range(BLOCK_B):
        a = tl.sum(tl.where(offs_b[:, None] == b, n1, 0.0), axis=0)
        c = tl.sum(tl.where(offs_b[:, None] == b, n2, 0.0), axis=0)
        acc += a[:, None] * c[None, :]

    full = mf[:, None] & mf[None, :]
    sq = tl.where(full, acc * acc, 0.0)
    total = tl.sum(tl.sum(sq, axis=1), axis=0)
    tl.store(out_ptr, total / (F * F))


class DiffLossNew(nn.Module):
    def __init__(self):
        super(DiffLossNew, self).__init__()

    def forward(self, input1, input2):
        B = input1.size(0)
        input1 = input1.reshape(B, -1).contiguous()
        input2 = input2.reshape(B, -1).contiguous()
        F = input1.size(1)
        out = torch.empty((), device=input1.device, dtype=torch.float32)
        BLOCK_B = triton.next_power_of_2(B)
        BLOCK_F = triton.next_power_of_2(F)
        _fused_kernel[(1,)](input1, input2, out, B, F,
                            BLOCK_B=BLOCK_B, BLOCK_F=BLOCK_F, num_warps=8)
        return out.to(input1.dtype)
