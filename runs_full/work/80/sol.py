import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, L_in, L_out,
    OC, ICG, OCG,
    S, D, P,
    sx_b, sx_c, sw_oc, sw_ic,
    so_b, so_c,
    HAS_BIAS: tl.constexpr,
    KSZ: tl.constexpr, ICG_C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // OC
    oc = pid % OC
    pid_t = tl.program_id(1)

    offs_t = pid_t * BLOCK + tl.arange(0, BLOCK)
    mask_t = offs_t < L_out

    g = oc // OCG
    ic_start = g * ICG

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for ic in range(ICG_C):
        x_base = x_ptr + b * sx_b + (ic_start + ic) * sx_c
        w_base = w_ptr + oc * sw_oc + ic * sw_ic
        for kk in range(KSZ):
            w = tl.load(w_base + kk).to(tl.float32)
            in_pos = offs_t * S + kk * D - P
            valid = mask_t & (in_pos >= 0) & (in_pos < L_in)
            x = tl.load(x_base + in_pos, mask=valid, other=0.0).to(tl.float32)
            acc += w * x

    if HAS_BIAS:
        acc += tl.load(b_ptr + oc).to(tl.float32)

    out_base = out_ptr + b * so_b + oc * so_c
    tl.store(out_base + offs_t, acc, mask=mask_t)


class CasualConv1dNew(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        dilation=1, groups=1, bias=True):
        super(CasualConv1dNew, self).__init__()
        self.dilation = dilation
        padding = dilation * (kernel_size - 1)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)

    def forward(self, input):
        x = input.contiguous()
        w = self.conv1d.weight
        bias = self.conv1d.bias
        B, C, L_in = x.shape
        OC = w.shape[0]
        ICG = w.shape[1]
        KSZ = w.shape[2]
        groups = C // ICG
        OCG = OC // groups
        S = self.conv1d.stride[0]
        D = self.conv1d.dilation[0]
        P = self.conv1d.padding[0]

        L_full = (L_in + 2 * P - D * (KSZ - 1) - 1) // S + 1
        L_out = L_full - self.dilation
        if L_out < 0:
            L_out = 0

        out = torch.empty((B, OC, L_out), device=x.device, dtype=x.dtype)
        if L_out == 0:
            return out

        BLOCK = 128
        grid = (B * OC, triton.cdiv(L_out, BLOCK))
        _causal_conv1d_kernel[grid](
            x, w, bias if bias is not None else x, out,
            B, L_in, L_out,
            OC, ICG, OCG,
            S, D, P,
            x.stride(0), x.stride(1), w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            HAS_BIAS=bias is not None,
            KSZ=KSZ, ICG_C=ICG,
            BLOCK=BLOCK, num_warps=4,
        )
        return out
