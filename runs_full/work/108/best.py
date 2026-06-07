import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gated_conv1d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    Cin, Cout, L, K, D, P,
    stride_xb, stride_xc, stride_xl,
    stride_wo, stride_wi, stride_wk,
    stride_ob, stride_oc, stride_ol,
    HAS_BIAS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)   # over B*Cout
    pid_t = tl.program_id(1)
    b = pid // Cout
    co = pid % Cout
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < L
    acc_mask = tl.zeros((BLOCK_T,), tl.float32)
    acc_out = tl.zeros((BLOCK_T,), tl.float32)
    for ci in range(Cin):
        for k in range(K):
            in_idx = offs_t + k * D - P
            in_mask = mask_t & (in_idx >= 0) & (in_idx < L)
            x = tl.load(x_ptr + b * stride_xb + ci * stride_xc + in_idx * stride_xl,
                        mask=in_mask, other=0.0)
            wm = tl.load(w_ptr + co * stride_wo + ci * stride_wi + k * stride_wk)
            wo = tl.load(w_ptr + (co + Cout) * stride_wo + ci * stride_wi + k * stride_wk)
            acc_mask += x * wm
            acc_out += x * wo
    if HAS_BIAS:
        acc_mask += tl.load(b_ptr + co)
        acc_out += tl.load(b_ptr + co + Cout)
    res = acc_out * tl.sigmoid(acc_mask)
    tl.store(out_ptr + b * stride_ob + co * stride_oc + offs_t * stride_ol,
             res, mask=mask_t)


class MaskedConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 groups=1, bias=True, causal=True):
        if causal:
            padding = (kernel_size - 1) * dilation
        else:
            padding = (kernel_size - 1) * dilation // 2
        super(MaskedConv1d, self).__init__(in_channels, out_channels,
            kernel_size, stride=1, padding=padding, dilation=dilation,
            groups=groups, bias=bias)


class GatedConv1dNew(MaskedConv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 groups=1, bias=True, causal=True):
        super(GatedConv1dNew, self).__init__(in_channels, 2 * out_channels,
            kernel_size, dilation, groups, bias, causal)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = inputs.contiguous()
        B, Cin, L = x.shape
        Cout = self.out_channels // 2
        K = self.kernel_size[0]
        D = self.dilation[0]
        P = self.padding[0]
        w = self.weight.contiguous()
        out = torch.empty((B, Cout, L), device=x.device, dtype=x.dtype)
        BLOCK_T = 128
        grid = (B * Cout, triton.cdiv(L, BLOCK_T))
        has_bias = self.bias is not None
        b_ptr = self.bias if has_bias else x
        _gated_conv1d_kernel[grid](
            x, w, b_ptr, out,
            Cin, Cout, L, K, D, P,
            x.stride(0), x.stride(1), x.stride(2),
            w.stride(0), w.stride(1), w.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            HAS_BIAS=has_bias, BLOCK_T=BLOCK_T, num_warps=4,
        )
        return out
