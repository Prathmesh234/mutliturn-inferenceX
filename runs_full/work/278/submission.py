import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(idepth_ptr, image_ptr, out_ptr,
                  N, Cimg, Cdep, H, W,
                  NX, NY,
                  BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
    p = tl.arange(0, BLOCK_P)
    c = tl.arange(0, BLOCK_C)
    HW = H * W
    sn_i = Cimg * HW
    sn_d = Cdep * HW
    m_img = c < Cimg
    m_dep = c < Cdep

    # ---- X direction ----
    Wm = W - 1
    wx = p % Wm
    tx = p // Wm
    hx = tx % H
    nx = tx // H
    base_x = nx * sn_i + hx * W + wx          # (P,)
    img_off = base_x[:, None] + c[None, :] * HW
    mp_x = (p < NX)[:, None]
    a = tl.load(image_ptr + img_off, mask=mp_x & m_img[None, :], other=0.0)
    b = tl.load(image_ptr + img_off + 1, mask=mp_x & m_img[None, :], other=0.0)
    wsum = tl.sum(tl.abs(a - b), axis=1)
    weight = tl.exp(-wsum / Cimg)             # (P,)
    base_xd = nx * sn_d + hx * W + wx
    dep_off = base_xd[:, None] + c[None, :] * HW
    da = tl.load(idepth_ptr + dep_off, mask=mp_x & m_dep[None, :], other=0.0)
    db = tl.load(idepth_ptr + dep_off + 1, mask=mp_x & m_dep[None, :], other=0.0)
    smx = tl.abs((da - db) * weight[:, None])
    sx = tl.sum(tl.where(mp_x & m_dep[None, :], smx, 0.0))

    # ---- Y direction ----
    Hm = H - 1
    wy = p % W
    ty = p // W
    hy = ty % Hm
    ny = ty // Hm
    base_y = ny * sn_i + hy * W + wy
    img_offy = base_y[:, None] + c[None, :] * HW
    mp_y = (p < NY)[:, None]
    ay = tl.load(image_ptr + img_offy, mask=mp_y & m_img[None, :], other=0.0)
    by = tl.load(image_ptr + img_offy + W, mask=mp_y & m_img[None, :], other=0.0)
    wsumy = tl.sum(tl.abs(ay - by), axis=1)
    weighty = tl.exp(-wsumy / Cimg)
    base_yd = ny * sn_d + hy * W + wy
    dep_offy = base_yd[:, None] + c[None, :] * HW
    day = tl.load(idepth_ptr + dep_offy, mask=mp_y & m_dep[None, :], other=0.0)
    dby = tl.load(idepth_ptr + dep_offy + W, mask=mp_y & m_dep[None, :], other=0.0)
    smy = tl.abs((day - dby) * weighty[:, None])
    sy = tl.sum(tl.where(mp_y & m_dep[None, :], smy, 0.0))

    cx = NX * Cdep
    cy = NY * Cdep
    tl.store(out_ptr, sx / cx + sy / cy)


class InverseDepthSmoothnessLossNew(nn.Module):
    def __init__(self) -> None:
        super(InverseDepthSmoothnessLossNew, self).__init__()

    def forward(self, idepth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        idepth = idepth.contiguous()
        image = image.contiguous()
        N, Cdep, H, W = idepth.shape
        Cimg = image.shape[1]
        BLOCK_C = triton.next_power_of_2(max(Cimg, Cdep))
        NX = N * H * (W - 1)
        NY = N * (H - 1) * W
        BLOCK_P = triton.next_power_of_2(max(NX, NY))

        out = torch.empty(1, device=idepth.device, dtype=torch.float32)
        _fused_kernel[(1,)](idepth, image, out, N, Cimg, Cdep, H, W, NX, NY,
                            BLOCK_P=BLOCK_P, BLOCK_C=BLOCK_C, num_warps=2)
        return out.to(idepth.dtype).reshape(())
