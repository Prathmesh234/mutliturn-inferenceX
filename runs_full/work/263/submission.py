import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _group_norm_kernel(x_ptr, out_ptr, w_ptr, b_ptr, group_size, sp,
                       cpg, num_groups, eps, HAS_AFFINE: tl.constexpr,
                       BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    g = pid % num_groups
    base = pid * group_size
    offs = tl.arange(0, BLOCK)
    mask = offs < group_size
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    n = group_size
    mean = tl.sum(x, axis=0) / n
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n
    rstd = 1.0 / tl.sqrt(var + eps)
    y = xc * rstd
    if HAS_AFFINE:
        ch = g * cpg + (offs // sp)
        w = tl.load(w_ptr + ch, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + ch, mask=mask, other=0.0).to(tl.float32)
        y = y * w + b
    tl.store(out_ptr + base + offs, y, mask=mask)


class Fp32GroupNormNew(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        x = input.float()
        if not x.is_contiguous():
            x = x.contiguous()
        N = x.shape[0]
        C = x.shape[1]
        G = self.num_groups
        sp = x.numel() // (N * C)
        cpg = C // G
        group_size = cpg * sp
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(group_size)
        has_affine = self.weight is not None
        w = self.weight if has_affine else x
        b = self.bias if has_affine else x
        _group_norm_kernel[(N * G,)](x, out, w, b, group_size, sp, cpg, G,
                                     self.eps, has_affine, BLOCK=BLOCK,
                                     num_warps=1)
        return out.type_as(input)
