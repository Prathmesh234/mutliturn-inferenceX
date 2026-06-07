import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _masked_mse_kernel(pred_ptr, target_ptr, mask_ptr, out_ptr,
                       n_elements, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    p = tl.load(pred_ptr + offs, mask=mask, other=0.0)
    t = tl.load(target_ptr + offs, mask=mask, other=0.0)
    m = tl.load(mask_ptr + offs, mask=mask, other=0.0)
    diff = p * m - t
    se = tl.sum(diff * diff, axis=0)
    ms = tl.sum(m, axis=0)
    tl.store(out_ptr, se / ms)


class MaskedMSELossNew(nn.Module):
    def __init__(self):
        super(MaskedMSELossNew, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):
        n = pred.numel()
        out = torch.empty((), device=pred.device, dtype=torch.float32)
        BLOCK_SIZE = triton.next_power_of_2(n)
        _masked_mse_kernel[(1,)](pred, target, mask, out, n,
                                 BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
        return out
