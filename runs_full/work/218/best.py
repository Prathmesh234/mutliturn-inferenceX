import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(pred_ptr, true_ptr, weight_ptr, out_ptr, M, L, S,
                  C: tl.constexpr, B: tl.constexpr, P_BLOCK: tl.constexpr):
    offs_p = tl.arange(0, P_BLOCK)
    pmask = offs_p < L
    n = offs_p // S
    s = offs_p % S
    base_p = n * C * S + s

    maxp = tl.full((P_BLOCK,), -float('inf'), tl.float32)
    for c in tl.static_range(C):
        pc = tl.load(pred_ptr + base_p + c * S, mask=pmask, other=-float('inf'))
        maxp = tl.maximum(maxp, pc)
    se = tl.zeros((P_BLOCK,), tl.float32)
    for c in tl.static_range(C):
        pc = tl.load(pred_ptr + base_p + c * S, mask=pmask, other=0.0)
        se += tl.exp(pc - maxp)
    lse = maxp + tl.log(se)

    maxt = tl.full((P_BLOCK,), -float('inf'), tl.float32)
    tgt = tl.zeros((P_BLOCK,), tl.int32)
    for c in tl.static_range(C):
        tc = tl.load(true_ptr + base_p + c * S, mask=pmask, other=-float('inf'))
        cond = tc > maxt
        tgt = tl.where(cond, c, tgt)
        maxt = tl.where(cond, tc, maxt)

    pred_t = tl.zeros((P_BLOCK,), tl.float32)
    for c in tl.static_range(C):
        pc = tl.load(pred_ptr + base_p + c * S, mask=pmask, other=0.0)
        pred_t = tl.where(tgt == c, pc, pred_t)

    loss = lse - pred_t

    wsum = tl.zeros((P_BLOCK,), tl.float32)
    for b in tl.static_range(B):
        wsum += tl.load(weight_ptr + b * L + offs_p, mask=pmask, other=0.0)

    total = tl.sum(tl.where(pmask, loss * wsum, 0.0), axis=0)
    res = total / M * 10.0
    res = tl.minimum(tl.maximum(res, 0.0), 20.0)
    tl.store(out_ptr, res)


class WeightedCrossEntropyLossNew(nn.Module):

    def __init__(self):
        super(WeightedCrossEntropyLossNew, self).__init__()
        self.bce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, y_pred, y_true, weight):
        y_pred = y_pred.contiguous()
        y_true = y_true.contiguous()
        weight = weight.contiguous()
        N = y_pred.shape[0]
        C = y_pred.shape[1]
        S = y_pred.numel() // (N * C)
        L = N * S
        M = weight.numel()
        B = M // L
        out = torch.empty((), device=y_pred.device, dtype=torch.float32)
        P_BLOCK = triton.next_power_of_2(L)
        _fused_kernel[(1,)](y_pred, y_true, weight, out, M, L, S,
                            C=C, B=B, P_BLOCK=P_BLOCK, num_warps=1)
        return out
