import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv1d_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                   B, IC, OC, L,
                   padding, dilation, OCG,
                   HAS_BIAS: tl.constexpr, K: tl.constexpr,
                   ICG: tl.constexpr, BLOCK_T: tl.constexpr):
    pid = tl.program_id(0)  # over B*OC
    b = pid // OC
    oc = pid % OC
    g = oc // OCG
    t = tl.arange(0, BLOCK_T)
    mask_t = t < L
    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for icg in tl.static_range(ICG):
        ic = g * ICG + icg
        x_base = b * IC * L + ic * L
        w_base = oc * ICG * K + icg * K
        for k in tl.static_range(K):
            in_pos = t - padding + k * dilation
            valid = (in_pos >= 0) & (in_pos < L) & mask_t
            xv = tl.load(x_ptr + x_base + in_pos, mask=valid, other=0.0)
            wv = tl.load(w_ptr + w_base + k)
            acc += xv * wv
    if HAS_BIAS:
        acc += tl.load(b_ptr + oc)
    out_off = b * OC * L + oc * L + t
    tl.store(out_ptr + out_off, acc, mask=mask_t)


class MaskedConv1dNew(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 groups=1, bias=True, causal=True):
        if causal:
            padding = (kernel_size - 1) * dilation
        else:
            padding = (kernel_size - 1) * dilation // 2
        super(MaskedConv1dNew, self).__init__(in_channels, out_channels,
                                              kernel_size, stride=1, padding=padding,
                                              dilation=dilation, groups=groups, bias=bias)

    def forward(self, inputs):
        x = inputs.contiguous()
        B, IC, L = x.shape
        OC = self.out_channels
        K = self.kernel_size[0]
        ICG = IC // self.groups
        OCG = OC // self.groups
        pad = self.padding[0]
        dil = self.dilation[0]
        out = torch.empty((B, OC, L), device=x.device, dtype=x.dtype)
        BLOCK_T = triton.next_power_of_2(L)
        w = self.weight.contiguous()
        has_bias = self.bias is not None
        bptr = self.bias if has_bias else x
        grid = (B * OC,)
        _conv1d_kernel[grid](x, w, bptr, out,
                             B, IC, OC, L, pad, dil, OCG,
                             HAS_BIAS=has_bias, K=K, ICG=ICG, BLOCK_T=BLOCK_T,
                             num_warps=4)
        return out
