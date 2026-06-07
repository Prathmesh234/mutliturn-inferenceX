import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _bce_kernel(yp_ptr, yt_ptr, out_ptr, inner, C, per_slice, total,
                BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < total
    x = tl.load(yp_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(yt_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    loss = tl.maximum(x, 0.0) - x * z + tl.log(1.0 + tl.exp(-tl.abs(x)))
    k = (offs // inner) % C
    w = tl.where(k == 0, 0.1, tl.where(k == 1, 0.5, 0.3))
    loss = tl.where(mask, loss * w, 0.0)
    s = tl.sum(loss, axis=0)
    tl.store(out_ptr, s / per_slice * 100.0)


class BCELossNew(nn.Module):

    def __init__(self):
        super(BCELossNew, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true, weights=None):
        y_pred = y_pred.contiguous()
        y_true = y_true.contiguous()
        C = y_pred.shape[1]
        inner = 1
        for d in y_pred.shape[2:]:
            inner *= d
        total = y_pred.numel()
        per_slice = total // C
        out = torch.empty((), device=y_pred.device, dtype=torch.float32)
        BLOCK_SIZE = triton.next_power_of_2(total)
        _bce_kernel[(1,)](y_pred, y_true, out, inner, C, per_slice, total,
                          BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out.to(y_pred.dtype)
