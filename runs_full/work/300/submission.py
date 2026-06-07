import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv_relu_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, H, W,
    P_h, P_w,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wh, stride_ww,
    C_IN: tl.constexpr, C_OUT: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr, PAD: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    pw = pid % P_w
    tmp = pid // P_w
    ph = tmp % P_h
    n = tmp // P_h

    co = tl.arange(0, BLOCK_CO)
    co_mask = co < C_OUT
    bias = tl.load(b_ptr + co, mask=co_mask, other=0.0)

    maxval = tl.full((BLOCK_CO,), -float('inf'), tl.float32)

    for dh in range(2):
        for dw in range(2):
            oh = ph * 2 + dh
            ow = pw * 2 + dw
            acc = tl.zeros((BLOCK_CO,), tl.float32)
            for ci in range(C_IN):
                for kh in range(KH):
                    ih = oh + kh - PAD
                    for kw in range(KW):
                        iw = ow + kw - PAD
                        valid = (ih >= 0) and (ih < H) and (iw >= 0) and (iw < W)
                        if valid:
                            xval = tl.load(x_ptr + n * stride_xn + ci * stride_xc
                                           + ih * stride_xh + iw * stride_xw)
                            wvec = tl.load(w_ptr + co * stride_wo + ci * stride_wc
                                           + kh * stride_wh + kw * stride_ww,
                                           mask=co_mask, other=0.0)
                            acc += xval * wvec
            acc += bias
            acc = tl.maximum(acc, 0.0)
            maxval = tl.maximum(maxval, acc)

    out_base = n * (C_OUT * P_h * P_w) + co * (P_h * P_w) + ph * P_w + pw
    tl.store(out_ptr + out_base, maxval, mask=co_mask)


class ConvBlockNew(nn.Module):
    def __init__(self, nb_in, nb_out):
        super(ConvBlockNew, self).__init__()
        self.convolution = nn.Conv2d(in_channels=nb_in, out_channels=nb_out,
            kernel_size=5, stride=1, padding=2)
        self.ReLU = nn.ReLU()
        self.MaxPooling = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = x.contiguous()
        N, C_in, H, W = x.shape
        C_out = self.convolution.out_channels
        H_out, W_out = H, W
        P_h, P_w = H_out // 2, W_out // 2
        w = self.convolution.weight
        b = self.convolution.bias
        out = torch.empty((N, C_out, P_h, P_w), device=x.device, dtype=x.dtype)
        BLOCK_CO = triton.next_power_of_2(C_out)
        grid = (N * P_h * P_w,)
        _conv_relu_pool_kernel[grid](
            x, w, b, out,
            N, H, W, P_h, P_w,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            C_IN=C_in, C_OUT=C_out, KH=5, KW=5, PAD=2,
            BLOCK_CO=BLOCK_CO, num_warps=2,
        )
        return out
