import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_out, L,
    C_in: tl.constexpr, K: tl.constexpr, DIL: tl.constexpr, P: tl.constexpr,
    HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr,
):
    n = tl.program_id(0)
    t = tl.arange(0, BLOCK)
    t_mask = t < L
    x_base = n * C_in * L

    for co in tl.static_range(C_out):
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        w_base = co * C_in * K
        for ci in tl.static_range(C_in):
            for k in tl.static_range(K):
                in_idx = t + k * DIL - P
                m = t_mask & (in_idx >= 0) & (in_idx < L)
                xv = tl.load(x_ptr + x_base + ci * L + in_idx, mask=m, other=0.0)
                wv = tl.load(w_ptr + w_base + ci * K + k)
                acc += xv * wv
        if HAS_BIAS:
            acc += tl.load(b_ptr + co)
        out_base = (n * C_out + co) * L
        tl.store(out_ptr + out_base + t, acc, mask=t_mask)


class CausalConv1dNew(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1,
        **kwargs):
        super(CausalConv1dNew, self).__init__(in_channels, out_channels,
            kernel_size, padding=dilation * (kernel_size - 1), dilation=
            dilation, **kwargs)

    def forward(self, input):
        input = input.contiguous()
        N, C_in, L = input.shape
        C_out = self.out_channels
        K = self.kernel_size[0]
        DIL = self.dilation[0]
        P = self.padding[0]
        out = torch.empty((N, C_out, L), device=input.device, dtype=input.dtype)
        w = self.weight.contiguous()
        b = self.bias if self.bias is not None else input
        BLOCK = 64
        grid = (N,)
        _causal_conv1d_kernel[grid](
            input, w, b, out,
            N, C_out, L,
            C_in=C_in, K=K, DIL=DIL, P=P,
            HAS_BIAS=self.bias is not None, BLOCK=BLOCK,
            num_warps=1,
        )
        return out
