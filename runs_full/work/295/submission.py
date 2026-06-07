import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _bce_kernel(mask_ptr, tmask_ptr, label_ptr, tlabel_ptr, out_ptr,
                n, beta, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(mask_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(tmask_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bce_m = -(y * tl.maximum(tl.log(x), -100.0) + (1.0 - y) * tl.maximum(tl.log(1.0 - x), -100.0))
    bce_m = tl.where(mask, bce_m, 0.0)

    a = tl.load(label_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(tlabel_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bce_l = -(b * tl.maximum(tl.log(a), -100.0) + (1.0 - b) * tl.maximum(tl.log(1.0 - a), -100.0))
    bce_l = tl.where(mask, bce_l, 0.0)

    pixel = tl.sum(bce_m) / n
    binary = tl.sum(bce_l) / n
    tl.store(out_ptr, pixel * beta + binary * (1.0 - beta))


class PixWiseBCELossNew(nn.Module):
    def __init__(self, beta=0.5):
        super().__init__()
        self.criterion = nn.BCELoss()
        self.beta = beta

    def forward(self, net_mask, net_label, target_mask, target_label):
        n = net_mask.numel()
        out = torch.empty((), device=net_mask.device, dtype=torch.float32)
        BLOCK_SIZE = triton.next_power_of_2(n)
        _bce_kernel[(1,)](net_mask, target_mask, net_label, target_label, out,
                          n, self.beta, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
        return out.to(net_mask.dtype)
