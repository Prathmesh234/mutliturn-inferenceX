import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _geo_kernel(z_ptr, vn_ptr, un_ptr, h_ptr, w_ptr, ch_ptr, cw_ptr,
                fh_ptr, fw_ptr, out_ptr, n_per, C, HW, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_per
    z = tl.load(z_ptr + offs, mask=mask)
    vn = tl.load(vn_ptr + offs, mask=mask)
    un = tl.load(un_ptr + offs, mask=mask)
    h = tl.load(h_ptr + offs, mask=mask)
    w = tl.load(w_ptr + offs, mask=mask)
    ch = tl.load(ch_ptr + offs, mask=mask)
    cw = tl.load(cw_ptr + offs, mask=mask)
    fh = tl.load(fh_ptr + offs, mask=mask)
    fw = tl.load(fw_ptr + offs, mask=mask)
    x = z * (0.5 * h * (vn + 1.0) - ch) / fh
    y = z * (0.5 * w * (un + 1.0) - cw) / fw
    per_batch = C * HW
    b = offs // per_batch
    rem = offs % per_batch
    base = b * (3 * per_batch) + rem
    tl.store(out_ptr + base, x, mask=mask)
    tl.store(out_ptr + base + per_batch, y, mask=mask)
    tl.store(out_ptr + base + 2 * per_batch, z, mask=mask)


class GeometryFeatureNew(nn.Module):
    def __init__(self):
        super(GeometryFeatureNew, self).__init__()

    def forward(self, z, vnorm, unorm, h, w, ch, cw, fh, fw):
        z = z.contiguous(); vnorm = vnorm.contiguous(); unorm = unorm.contiguous()
        h = h.contiguous(); w = w.contiguous(); ch = ch.contiguous()
        cw = cw.contiguous(); fh = fh.contiguous(); fw = fw.contiguous()
        N, C, H, W = z.shape
        HW = H * W
        n_per = z.numel()
        out = torch.empty((N, 3 * C, H, W), device=z.device, dtype=z.dtype)
        BLOCK = 1024
        grid = (triton.cdiv(n_per, BLOCK),)
        _geo_kernel[grid](z, vnorm, unorm, h, w, ch, cw, fh, fw, out,
                          n_per, C, HW, BLOCK=BLOCK, num_warps=4)
        return out
