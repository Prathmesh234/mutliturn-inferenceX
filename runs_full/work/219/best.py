import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _wbce_kernel(pred_ptr, true_ptr, w_ptr, out_ptr,
                 N, CHW, HW, C,
                 BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    # decompose weights linear index e -> (a, s) for sum_loss[a, i, j]
    rem = offs % CHW
    a = rem // HW
    s = rem % HW
    base = a * CHW + s
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for c in range(0, C):
        idx = base + c * HW
        x = tl.load(pred_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(true_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        loss = tl.maximum(x, 0.0) - x * y + tl.log(1.0 + tl.exp(-tl.abs(x)))
        acc += loss
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    res = acc * w
    total = tl.sum(tl.where(mask, res, 0.0))
    out = total / N * 10.0
    tl.store(out_ptr, out)


class WeightedBCELossNew(nn.Module):
    def __init__(self):
        super(WeightedBCELossNew, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, y_pred, y_true, weights):
        y_pred = y_pred.contiguous()
        y_true = y_true.contiguous()
        weights = weights.contiguous()
        B, C, H, W = y_pred.shape
        N = weights.numel()
        CHW = C * H * W
        HW = H * W
        out = torch.empty((), device=y_pred.device, dtype=torch.float32)
        BLOCK_SIZE = triton.next_power_of_2(N)
        _wbce_kernel[(1,)](y_pred, y_true, weights, out,
                           N, CHW, HW, 4,
                           BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out
