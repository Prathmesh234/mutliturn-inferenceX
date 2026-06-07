import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _norm_kernel(x_ptr, out_ptr, outer, R, C, eps,
                 BLOCK_O: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK_O + tl.arange(0, BLOCK_O)
    om = o < outer
    n = o // R
    r = o % R
    c = tl.arange(0, BLOCK_C)
    cm = c < C
    addr = (n * (C * R))[:, None] + c[None, :] * R + r[:, None]
    mask = om[:, None] & cm[None, :]
    x = tl.load(x_ptr + addr, mask=mask, other=0.0)
    sq = x * x
    mean = tl.sum(sq, axis=1) / C
    scale = (1.0 / tl.sqrt(mean + eps))[:, None]
    tl.store(out_ptr + addr, x * scale, mask=mask)


class NormalizationLayerNew(nn.Module):

    def __init__(self):
        super(NormalizationLayerNew, self).__init__()

    def forward(self, x, epsilon=1e-08):
        out = torch.empty_like(x)
        C = x.shape[1]
        R = 1
        for d in x.shape[2:]:
            R *= d
        outer = x.shape[0] * R
        BLOCK_C = triton.next_power_of_2(C)
        BLOCK_O = 128
        grid = (triton.cdiv(outer, BLOCK_O),)
        _norm_kernel[grid](x, out, outer, R, C, float(epsilon),
                           BLOCK_O=BLOCK_O, BLOCK_C=BLOCK_C, num_warps=4)
        return out
