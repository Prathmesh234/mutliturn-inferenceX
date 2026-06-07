import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _focal_kernel(logits_ptr, labels_ptr, out_ptr, n_elements,
                  gamma, pos_weight, neg_weight,
                  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(logits_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(labels_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    absx = tl.abs(x)
    # softplus(-x) = max(-x,0) + log1p(exp(-|x|))
    sp_negx = tl.maximum(-x, 0.0) + tl.log(1.0 + tl.exp(-absx))
    # ce = max(x,0) - x*z + log1p(exp(-|x|))
    ce = tl.maximum(x, 0.0) - x * z + tl.log(1.0 + tl.exp(-absx))

    modulator = tl.exp(-gamma * z * x - gamma * sp_negx)
    alpha = z * pos_weight + (1.0 - z) * neg_weight
    wl = alpha * modulator * ce

    wl = tl.where(mask, wl, 0.0)
    block_sum = tl.sum(wl)
    tl.atomic_add(out_ptr, block_sum)


class Focal_lossNew(nn.Module):
    def __init__(self, gamma=0):
        super().__init__()
        self.cross_entropy = nn.BCEWithLogitsLoss(reduction='none')
        self.gamma = gamma

    def forward(self, logits, labels, pos_weight=1, neg_weight=1):
        logits = logits.contiguous()
        labels = labels.contiguous()
        n = logits.numel()
        out = torch.zeros(1, device=logits.device, dtype=torch.float32)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _focal_kernel[grid](logits, labels, out, n,
                            float(self.gamma), float(pos_weight), float(neg_weight),
                            BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return (out / n).reshape(())
