import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _triplet_kernel(a_ptr, p_ptr, n_ptr, out_ptr, n_rows, inv_n, D, margin,
                    BLOCK_M: tl.constexpr, D_POW2: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_rows
    cols = tl.arange(0, D_POW2)
    col_mask = cols < D
    offs = rows[:, None] * D + cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]

    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    p = tl.load(p_ptr + offs, mask=mask, other=0.0)
    nn_ = tl.load(n_ptr + offs, mask=mask, other=0.0)

    dp = a - p
    dn = a - nn_
    pos_d = tl.sum(dp * dp, axis=1)
    neg_d = tl.sum(dn * dn, axis=1)
    val = margin + pos_d - neg_d
    val = tl.where(val > 0.0, val, 0.0)
    val = tl.where(row_mask, val, 0.0)
    s = tl.sum(val, axis=0) * inv_n
    tl.atomic_add(out_ptr, s)


class TripletLossNew(nn.Module):
    def __init__(self):
        super(TripletLossNew, self).__init__()
        self.margin = 0.5

    def forward(self, anchor, pos, neg):
        anchor = anchor.contiguous()
        pos = pos.contiguous()
        neg = neg.contiguous()
        D = anchor.shape[-1]
        n_rows = anchor.numel() // D
        out = torch.zeros([], device=anchor.device, dtype=torch.float32)
        D_POW2 = triton.next_power_of_2(D)
        BLOCK_M = 128
        grid = (triton.cdiv(n_rows, BLOCK_M),)
        _triplet_kernel[grid](anchor, pos, neg, out, n_rows, 1.0 / n_rows, D, self.margin,
                              BLOCK_M=BLOCK_M, D_POW2=D_POW2, num_warps=4)
        return out
