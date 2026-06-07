import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr, w3_ptr, w32_ptr, out_ptr, HW,
                  C: tl.constexpr, C1: tl.constexpr,
                  CB: tl.constexpr, C1B: tl.constexpr, HWB: tl.constexpr):
    n = tl.program_id(0)
    offs_c = tl.arange(0, CB)
    offs_c1 = tl.arange(0, C1B)
    offs_hw = tl.arange(0, HWB)
    mask_c = offs_c < C
    mask_c1 = offs_c1 < C1
    mask_hw = offs_hw < HW

    # load x[n] : [CB, HWB]
    xp = x_ptr + n * C * HW + offs_c[:, None] * HW + offs_hw[None, :]
    m = mask_c[:, None] & mask_hw[None, :]
    x = tl.load(xp, mask=m, other=0.0)
    s = tl.sum(x, axis=1)
    mean = s / HW
    d = tl.where(m, x - mean[:, None], 0.0)
    var = tl.sum(d * d, axis=1) / HW
    y = tl.sqrt(var) + mean            # [CB]
    y = tl.where(mask_c, y, 0.0)

    # z = relu(W3 @ y) : W3 [C1, C] -> [C1B, CB]
    w3 = tl.load(w3_ptr + offs_c1[:, None] * C + offs_c[None, :],
                 mask=mask_c1[:, None] & mask_c[None, :], other=0.0)
    z = tl.sum(w3 * y[None, :], axis=1)   # [C1B]
    z = tl.maximum(z, 0.0)
    z = tl.where(mask_c1, z, 0.0)

    # out = sigmoid(W32 @ z) : W32 [C, C1] -> [CB, C1B]
    w32 = tl.load(w32_ptr + offs_c[:, None] * C1 + offs_c1[None, :],
                  mask=mask_c[:, None] & mask_c1[None, :], other=0.0)
    o = tl.sum(w32 * z[None, :], axis=1)  # [CB]
    o = 1.0 / (1.0 + tl.exp(-o))
    tl.store(out_ptr + n * C + offs_c, o, mask=mask_c)


class LCCALayerNew(nn.Module):

    def __init__(self, channel):
        super(LCCALayerNew, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c3 = nn.Conv2d(channel, channel // 4, kernel_size=3, padding=1,
                            bias=False)
        self.c32 = nn.Conv2d(channel // 4, channel, kernel_size=3, padding=1,
                             bias=False)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, H, W = x.shape
        HW = H * W
        C1 = C // 4
        x = x.contiguous()
        W3c = self.c3.weight[:, :, 1, 1].contiguous()
        W32c = self.c32.weight[:, :, 1, 1].contiguous()
        out = torch.empty(N, C, device=x.device, dtype=x.dtype)
        _fused_kernel[(N,)](x, W3c, W32c, out, HW,
                            C=C, C1=C1,
                            CB=triton.next_power_of_2(C),
                            C1B=triton.next_power_of_2(C1),
                            HWB=triton.next_power_of_2(HW),
                            num_warps=4)
        return out.view(N, C, 1, 1)
