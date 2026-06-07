import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _padded_maxpool_kernel(
    x_ptr, out_ptr,
    NC, H, W, OH, OW,
    sh, sw, pad_top, pad_left, dh, dw,
    BLOCK: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_s = tl.program_id(1)
    sp = pid_s * BLOCK + tl.arange(0, BLOCK)
    mask = sp < (OH * OW)

    ow = sp % OW
    oh = sp // OW

    base = pid_nc * H
    h0 = oh * sh - pad_top
    w0 = ow * sw - pad_left

    acc = tl.full((BLOCK,), -float('inf'), tl.float32)
    for i in tl.static_range(KH):
        ih = tl.maximum(0, tl.minimum(h0 + i * dh, H - 1))
        rowbase = (base + ih) * W
        for j in tl.static_range(KW):
            iw = tl.maximum(0, tl.minimum(w0 + j * dw, W - 1))
            v = tl.load(x_ptr + rowbase + iw, mask=mask, other=-float('inf'))
            acc = tl.maximum(acc, v)

    out_base = pid_nc * (OH * OW)
    tl.store(out_ptr + out_base + sp, acc, mask=mask)


class PaddedMaxPool2dNew(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=(0, 0, 0, 0), dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        kh, kw = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        sh, sw = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        dh, dw = self.dilation if isinstance(self.dilation, tuple) else (self.dilation, self.dilation)
        pl, pr, pt, pb = self.padding

        N, C, H, W = x.shape
        Hp = H + pt + pb
        Wp = W + pl + pr
        OH = (Hp - dh * (kh - 1) - 1) // sh + 1
        OW = (Wp - dw * (kw - 1) - 1) // sw + 1

        out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
        NC = N * C
        OS = OH * OW
        BLOCK = triton.next_power_of_2(OS)
        grid = (NC, triton.cdiv(OS, BLOCK))
        _padded_maxpool_kernel[grid](
            x, out, NC, H, W, OH, OW,
            sh, sw, pt, pl, dh, dw,
            BLOCK=BLOCK, KH=kh, KW=kw, num_warps=4,
        )
        return out
