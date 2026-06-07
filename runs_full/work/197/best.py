import torch
import torch.nn as nn
import triton
import triton.language as tl


def centercrop(image, w, h):
    _nt, _ct, ht, wt = image.size()
    padw, padh = (wt - w) // 2, (ht - h) // 2
    if padw > 0 and padh > 0:
        image = image[:, :, padh:-padh, padw:-padw]
    return image


@triton.jit
def _dice_kernel(pred_ptr, true_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    p = tl.load(pred_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    t = tl.load(true_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    p = 1.0 / (1.0 + tl.exp(-p))
    inter = tl.sum(p * t)
    sp = tl.sum(p)
    st = tl.sum(t)
    score = (2.0 * inter + 1.0) / (st + sp + 1.0)
    tl.store(out_ptr, 1.0 - score)


class BDiceLossNew(nn.Module):

    def __init__(self):
        super(BDiceLossNew, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_pred, y_true, weight=None):
        _n, _ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h).contiguous()
        y_pred = y_pred.contiguous()
        n = y_pred.numel()
        out = torch.empty(1, device=y_pred.device, dtype=torch.float32)
        BLOCK_SIZE = triton.next_power_of_2(n)
        _dice_kernel[(1,)](y_pred, y_true, out, n,
                           BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out[0]
