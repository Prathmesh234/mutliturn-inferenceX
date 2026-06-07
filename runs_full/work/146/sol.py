import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IC, IH, IW, OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
    HAS_BIAS: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_boc = tl.program_id(0)
    pid_p = tl.program_id(1)
    b = pid_boc // OC
    oc = pid_boc % OC

    offs_p = pid_p * BLOCK + tl.arange(0, BLOCK)
    mask_p = offs_p < (OH * OW)
    oh = offs_p // OW
    ow = offs_p % OW

    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    x_base = b * IC * IH * IW
    w_base = oc * IC * KH * KW

    for ic in range(IC_C):
        for kh in range(KH):
            for kw in range(KW):
                ih = oh * stride_h - pad_h + kh * dil_h
                iw = ow * stride_w - pad_w + kw * dil_w
                valid = mask_p & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
                x_off = x_base + ic * IH * IW + ih * IW + iw
                xv = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                wv = tl.load(w_ptr + w_base + ic * KH * KW + kh * KW + kw)
                acc += xv * wv.to(tl.float32)

    if HAS_BIAS:
        acc += tl.load(b_ptr + oc).to(tl.float32)

    out_base = (b * OC + oc) * OH * OW
    tl.store(out_ptr + out_base + offs_p, acc, mask=mask_p)


class Conv2dTimeNew(nn.Conv2d):
    def __init__(self, in_channels, *args, **kwargs):
        super(Conv2dTimeNew, self).__init__(in_channels + 1, *args, **kwargs)

    def forward(self, t, x):
        t_img = torch.ones_like(x[:, :1, :, :]) * t
        t_and_x = torch.cat([t_img, x], 1).contiguous()

        B, IC, IH, IW = t_and_x.shape
        OC = self.out_channels
        KH, KW = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dil_h, dil_w = self.dilation

        OH = (IH + 2 * pad_h - dil_h * (KH - 1) - 1) // stride_h + 1
        OW = (IW + 2 * pad_w - dil_w * (KW - 1) - 1) // stride_w + 1

        out = torch.empty((B, OC, OH, OW), device=x.device, dtype=t_and_x.dtype)

        n_pix = OH * OW
        BLOCK = min(triton.next_power_of_2(n_pix), 256)
        if BLOCK < 1:
            BLOCK = 1
        grid = (B * OC, triton.cdiv(n_pix, BLOCK))

        w = self.weight.contiguous()
        has_bias = self.bias is not None
        b_ptr = self.bias if has_bias else t_and_x

        _conv2d_kernel[grid](
            t_and_x, w, b_ptr, out,
            B, IC, IH, IW, OC, OH, OW,
            KH, KW,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            has_bias,
            IC,
            BLOCK,
            num_warps=4,
        )
        return out
