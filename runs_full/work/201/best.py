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
def _bdice_kernel(yp_ptr, m1_ptr, w_ptr, out_ptr, n_elements,
                  BLOCK_SIZE: tl.constexpr):
    s_wmm = 0.0
    s_wm1 = 0.0
    s_wm2 = 0.0
    for start in range(0, n_elements, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        yp = tl.load(yp_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        m1 = tl.load(m1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        m2 = 1.0 / (1.0 + tl.exp(-yp))
        wm1 = w * m1
        s_wmm += tl.sum(wm1 * m2)
        s_wm1 += tl.sum(wm1)
        s_wm2 += tl.sum(w * m2)
    score = (2.0 * s_wmm + 1.0) / (s_wm1 + s_wm2 + 1.0)
    tl.store(out_ptr, 1.0 - score)


class WeightedBDiceLossNew(nn.Module):
    def __init__(self):
        super(WeightedBDiceLossNew, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_pred, y_true, weight):
        _n, _ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h).contiguous()
        weight = centercrop(weight, w, h).contiguous()
        y_pred = y_pred.contiguous()
        n = y_pred.numel()
        out = torch.empty((), device=y_pred.device, dtype=torch.float32)
        BLOCK_SIZE = 1024
        _bdice_kernel[(1,)](y_pred, y_true, weight, out, n,
                            BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
        return out
