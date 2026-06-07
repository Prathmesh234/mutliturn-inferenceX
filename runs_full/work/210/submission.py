import torch
import triton
import triton.language as tl


def centercrop(image, w, h):
    _nt, _ct, ht, wt = image.size()
    padw, padh = (wt - w) // 2, (ht - h) // 2
    if padw > 0 and padh > 0:
        image = image[:, :, padh:-padh, padw:-padw]
    return image


@triton.jit
def _wmce_kernel(yp_ptr, yt_ptr, w_ptr, out_ptr, n_pos, HW, inv,
                 C: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.arange(0, BLOCK)
    pmask = p < n_pos
    n = p // HW
    hw = p % HW
    c = tl.arange(0, C)
    off = (n * (C * HW) + hw)[:, None] + c[None, :] * HW
    mask = pmask[:, None]
    x = tl.load(yp_ptr + off, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=1)
    xs = tl.load(yp_ptr + off, mask=mask, other=0.0)
    s = tl.sum(tl.exp(xs - m[:, None]), axis=1)
    lse = m + tl.log(s)
    yt = tl.load(yt_ptr + off, mask=mask, other=0.0)
    wv = tl.load(w_ptr + off, mask=mask, other=0.0)
    acc = tl.sum(wv * (xs - lse[:, None]) * yt, axis=1)
    total = tl.sum(tl.where(pmask, acc, 0.0))
    tl.store(out_ptr, -total * inv)


class WeightedMCElossNew(torch.nn.Module):
    def __init__(self):
        super(WeightedMCElossNew, self).__init__()

    def forward(self, y_pred, y_true, weight):
        _n, _ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h).contiguous()
        weight = centercrop(weight, w, h).contiguous()
        y_pred = y_pred.contiguous()
        N, C, H, W = y_pred.size()
        HW = H * W
        n_pos = N * HW
        out = torch.empty(1, device=y_pred.device, dtype=torch.float32)
        BLOCK = triton.next_power_of_2(n_pos)
        _wmce_kernel[(1,)](y_pred, y_true, weight, out, n_pos, HW,
                           1.0 / n_pos, C=C, BLOCK=BLOCK, num_warps=4)
        return out.reshape([])
