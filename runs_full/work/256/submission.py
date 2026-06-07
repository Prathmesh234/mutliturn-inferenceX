import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _bce_blur_kernel(pred_ptr, true_ptr, out_ptr, n_elements, inv_alpha, inv_n,
                     BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    pred = tl.load(pred_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    true = tl.load(true_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    loss = tl.maximum(pred, 0.0) - pred * true + tl.log(1.0 + tl.exp(-tl.abs(pred)))
    sig = 1.0 / (1.0 + tl.exp(-pred))
    dx = sig - true
    alpha_factor = 1.0 - tl.exp((dx - 1.0) * inv_alpha)
    val = tl.where(mask, loss * alpha_factor, 0.0)
    tl.store(out_ptr, tl.sum(val, axis=0) * inv_n)


class BCEBlurWithLogitsLossNew(nn.Module):

    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLossNew, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha

    def forward(self, pred, true):
        n = pred.numel()
        out = torch.empty((), device=pred.device, dtype=pred.dtype)
        BLOCK_SIZE = triton.next_power_of_2(n)
        inv_alpha = 1.0 / (self.alpha + 0.0001)
        _bce_blur_kernel[(1,)](pred, true, out, n, inv_alpha, 1.0 / n,
                               BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
        return out
