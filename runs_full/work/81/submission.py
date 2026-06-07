import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _dense_kernel(
    in_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
    L, dilation,
    sB, sC, sL,
    wF, wC,
    oB, oC, oL,
    C: tl.constexpr, F: tl.constexpr, K: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    CF: tl.constexpr = C + F
    b = pid0 // CF
    ch = pid0 % CF

    offs_t = pid1 * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < L

    if ch < C:
        x = tl.load(in_ptr + b * sB + ch * sC + offs_t * sL, mask=mask_t)
        tl.store(out_ptr + b * oB + ch * oC + offs_t * oL, x, mask=mask_t)
    else:
        f = ch - C
        acc_f = tl.load(b1_ptr + f).to(tl.float32) + tl.zeros((BLOCK_T,), tl.float32)
        acc_g = tl.load(b2_ptr + f).to(tl.float32) + tl.zeros((BLOCK_T,), tl.float32)
        for ic in range(C):
            for k in range(K):
                in_idx = offs_t + dilation * (k - (K - 1))
                valid = mask_t & (in_idx >= 0) & (in_idx < L)
                xv = tl.load(in_ptr + b * sB + ic * sC + in_idx * sL, mask=valid, other=0.0).to(tl.float32)
                w1v = tl.load(w1_ptr + f * wF + ic * wC + k).to(tl.float32)
                w2v = tl.load(w2_ptr + f * wF + ic * wC + k).to(tl.float32)
                acc_f += w1v * xv
                acc_g += w2v * xv
        tf = (1.0 - tl.exp(-2.0 * acc_f)) / (1.0 + tl.exp(-2.0 * acc_f))
        sg = 1.0 / (1.0 + tl.exp(-acc_g))
        tl.store(out_ptr + b * oB + ch * oC + offs_t * oL, tf * sg, mask=mask_t)


class CasualConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, bias=True):
        super(CasualConv1d, self).__init__()
        self.dilation = dilation
        padding = dilation * (kernel_size - 1)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride, padding, dilation, groups, bias)

    def forward(self, input):
        out = self.conv1d(input)
        return out[:, :, :-self.dilation]


class DenseBlockNew(nn.Module):
    def __init__(self, in_channels, dilation, filters, kernel_size=2):
        super(DenseBlockNew, self).__init__()
        self.casualconv1 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)
        self.casualconv2 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)
        self.dilation = dilation

    def forward(self, input):
        input = input.contiguous()
        B, C, L = input.shape
        w1 = self.casualconv1.conv1d.weight
        b1 = self.casualconv1.conv1d.bias
        w2 = self.casualconv2.conv1d.weight
        b2 = self.casualconv2.conv1d.bias
        F, _, K = w1.shape
        out = torch.empty((B, C + F, L), device=input.device, dtype=input.dtype)

        BLOCK_T = triton.next_power_of_2(L)
        sB, sC, sL = input.stride()
        oB, oC, oL = out.stride()
        wF, wC, _ = w1.stride()

        grid = (B * (C + F), triton.cdiv(L, BLOCK_T))
        _dense_kernel[grid](
            input, w1, b1, w2, b2, out,
            L, self.dilation,
            sB, sC, sL, wF, wC, oB, oC, oL,
            C=C, F=F, K=K, BLOCK_T=BLOCK_T, num_warps=4,
        )
        return out
