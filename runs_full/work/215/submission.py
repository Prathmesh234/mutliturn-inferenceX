import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mcc_kernel(yp_ptr, yt_ptr, out_ptr, n_groups, C, inner, eps,
                BLOCK_G: tl.constexpr, BLOCK_C: tl.constexpr):
    g = tl.arange(0, BLOCK_G)
    c = tl.arange(0, BLOCK_C)
    n_idx = g // inner
    hw = g % inner
    offs = (n_idx * C * inner)[:, None] + c[None, :] * inner + hw[:, None]
    mask = (g < n_groups)[:, None] & (c < C)[None, :]
    xp = tl.load(yp_ptr + offs, mask=mask, other=-float('inf')).to(tl.float32)
    m = tl.max(xp, axis=1)
    e = tl.exp(xp - m[:, None])
    ssum = tl.sum(e, axis=1)
    yp = e / ssum[:, None]
    yp = tl.where(mask, yp, 0.0)
    yt = tl.load(yt_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    sx = tl.sum(yt)
    sy = tl.sum(yp)
    sx2 = tl.sum(yt * yt)
    sy2 = tl.sum(yp * yp)
    sxy = tl.sum(yt * yp)

    nf = (n_groups * C).to(tl.float32)
    x_mean = sx / nf
    y_mean = sy / nf
    x_var = (sx2 - sx * sx / nf) / (nf - 1)
    y_var = (sy2 - sy * sy / nf) / (nf - 1)
    x_std = tl.sqrt(x_var)
    y_std = tl.sqrt(y_var)
    sum_vxvy = sxy - sx * sy / nf
    sum_vx2 = sx2 - sx * sx / nf
    sum_vy2 = sy2 - sy * sy / nf
    pcc = sum_vxvy / (tl.sqrt(sum_vx2 + eps) * tl.sqrt(sum_vy2 + eps))
    ccc = 2 * pcc * x_std * y_std / (x_var + y_var + (y_mean - x_mean) * (y_mean - x_mean))
    res = (1 - ccc) * 10
    tl.store(out_ptr, res)


class MCCLossNew(nn.Module):

    def __init__(self, eps=1e-06):
        super(MCCLossNew, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true, w=None):
        y_pred = y_pred.contiguous()
        y_true = y_true.contiguous()
        N, C = y_pred.shape[0], y_pred.shape[1]
        inner = y_pred.numel() // (N * C)
        n_groups = N * inner
        out = torch.empty((), device=y_pred.device, dtype=torch.float32)
        BLOCK_G = triton.next_power_of_2(n_groups)
        BLOCK_C = triton.next_power_of_2(C)
        _mcc_kernel[(1,)](y_pred, y_true, out, n_groups, C, inner, self.eps,
                          BLOCK_G=BLOCK_G, BLOCK_C=BLOCK_C, num_warps=1)
        return out
