import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _bce_kernel(pred_ptr, true_ptr, out_ptr, M, S_c, C, BLOCK: tl.constexpr):
    acc = tl.zeros((BLOCK,), tl.float32)
    for c in tl.static_range(2):
        base = c * S_c
        cacc = tl.zeros((BLOCK,), tl.float32)
        for start in range(0, M, BLOCK):
            idx = start + tl.arange(0, BLOCK)
            mask = idx < M
            n = idx // S_c
            rem = idx % S_c
            gidx = n * C * S_c + base + rem
            x = tl.load(pred_ptr + gidx, mask=mask, other=0.0).to(tl.float32)
            z = tl.load(true_ptr + gidx, mask=mask, other=0.0).to(tl.float32)
            val = tl.maximum(x, 0.0) - x * z + tl.log(1.0 + tl.exp(-tl.abs(x)))
            cacc += tl.where(mask, val, 0.0)
        acc += cacc / M
    tl.store(out_ptr, tl.sum(acc))


class BCELoss2cNew(nn.Module):

    def __init__(self):
        super(BCELoss2cNew, self).__init__()
        self.bce0 = nn.BCEWithLogitsLoss()
        self.bce1 = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true, weights=None):
        y_pred = y_pred.contiguous()
        y_true = y_true.contiguous()
        N = y_pred.shape[0]
        C = y_pred.shape[1]
        S_c = 1
        for d in y_pred.shape[2:]:
            S_c *= d
        M = N * S_c
        out = torch.empty(1, device=y_pred.device, dtype=torch.float32)
        BLOCK = triton.next_power_of_2(M)
        _bce_kernel[(1,)](y_pred, y_true, out, M, S_c, C, BLOCK=BLOCK, num_warps=1)
        return out[0]
