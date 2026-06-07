import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _expand_kernel(x_ptr, out_ptr, n_elements,
                   N, C, H, W, Cout, Hout, Wout,
                   s: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    wo = offs % Wout
    t = offs // Wout
    ho = t % Hout
    t = t // Hout
    c = t % Cout
    n = t // Cout

    h = ho // s
    a = ho % s
    w = wo // s
    b = wo % s

    src_c = a * s * Cout + b * Cout + c
    src = ((n * C + src_c) * H + h) * W + w

    val = tl.load(x_ptr + src, mask=mask)
    tl.store(out_ptr + offs, val, mask=mask)


class ExpandNew(nn.Module):
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()
        s = self.gain
        Cout = C // (s * s)
        Hout = H * s
        Wout = W * s
        x = x.contiguous()
        out = torch.empty((N, Cout, Hout, Wout), device=x.device, dtype=x.dtype)
        n_elements = out.numel()
        BLOCK_SIZE = triton.next_power_of_2(n_elements)
        _expand_kernel[(1,)](x, out, n_elements,
                             N, C, H, W, Cout, Hout, Wout,
                             s=s, BLOCK_SIZE=BLOCK_SIZE, num_warps=2)
        return out
