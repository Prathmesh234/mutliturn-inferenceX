import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(
    x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
    N, C, H, W,
    BLOCK_HW: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    n = pid_nc // C
    oc = pid_nc % C

    offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    HW = H * W
    mask = offs < HW
    oh = offs // W
    ow = offs % W

    b2 = tl.load(b2_ptr + oc)
    out_base = (n * C + oc) * HW + offs
    idv = tl.load(x_ptr + out_base, mask=mask, other=0.0)
    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32) + b2 + idv

    for mc in range(C):
        b1 = tl.load(b1_ptr + mc)
        x_base = n * C * HW
        w1_base = mc * C * 9
        for kh in range(3):
            ph = oh + kh - 1
            for kw in range(3):
                pw = ow + kw - 1
                pvalid = mask & (ph >= 0) & (ph < H) & (pw >= 0) & (pw < W)
                c1 = tl.zeros((BLOCK_HW,), dtype=tl.float32) + b1
                for ic in range(C):
                    xb = x_base + ic * HW
                    wb = w1_base + ic * 9
                    for jh in range(3):
                        ih = ph + jh - 1
                        for jw in range(3):
                            iw = pw + jw - 1
                            vv = pvalid & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                            xv = tl.load(x_ptr + xb + ih * W + iw, mask=vv, other=0.0)
                            wv = tl.load(w1_ptr + wb + jh * 3 + jw)
                            c1 += xv * wv
                c1 = tl.where(c1 >= 0, c1, c1 * 0.2)
                c1 = tl.where(pvalid, c1, 0.0)
                w2v = tl.load(w2_ptr + (oc * C + mc) * 9 + kh * 3 + kw)
                acc += c1 * w2v

    tl.store(out_ptr + out_base, acc, mask=mask)


class _Residual_Block_SRNew(nn.Module):

    def __init__(self, num_ft):
        super(_Residual_Block_SRNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_ft, out_channels=num_ft,
            kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_ft, out_channels=num_ft,
            kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = x.contiguous()
        N, C, H, W = x.shape
        out = torch.empty_like(x)
        HW = H * W
        BLOCK_HW = min(1024, triton.next_power_of_2(HW))
        grid = (N * C, triton.cdiv(HW, BLOCK_HW))
        _fused_kernel[grid](
            x, self.conv1.weight.contiguous(), self.conv1.bias,
            self.conv2.weight.contiguous(), self.conv2.bias, out,
            N, C, H, W, BLOCK_HW=BLOCK_HW, num_warps=1,
        )
        return out
