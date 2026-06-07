import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr, out_ptr, n,
                  s0, s1, s2, s3,
                  as0, as1, as2, as3,
                  bs0, bs1, bs2, bs3,
                  Cg, eps, BLOCK: tl.constexpr, CG: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    i3 = idx % s3
    t = idx // s3
    i2 = t % s2
    t = t // s2
    i1 = t % s1
    t = t // s1
    i0 = t % s0
    a_off = i0 * as0 + i1 * as1 + i2 * as2 + i3 * as3
    b_off = i0 * bs0 + i1 * bs1 + i2 * bs2 + i3 * bs3

    a = tl.load(x_ptr + a_off, mask=mask)

    gb = (b_off // Cg) * Cg
    cj = tl.arange(0, CG)
    g_off = gb[:, None] + cj[None, :]
    g_mask = mask[:, None] & (cj[None, :] < Cg)
    g = tl.load(x_ptr + g_off, mask=g_mask, other=0.0).to(tl.float32)
    m2 = tl.sum(g * g, axis=1) / Cg

    out = a / tl.sqrt(m2 + eps)
    tl.store(out_ptr + idx, out, mask=mask)


def _pad(shape, strides, ndim):
    shape = [1] * (ndim - len(shape)) + list(shape)
    strides = [0] * (ndim - len(strides)) + list(strides)
    return shape, strides


class GroupScaling1DNew(nn.Module):
    """Scales inputs by the second moment for the entire layer."""

    def __init__(self, eps=1e-05, group_num=4):
        super(GroupScaling1DNew, self).__init__()
        self.eps = eps
        self.group_num = group_num

    def extra_repr(self):
        return f'eps={self.eps}, group={self.group_num}'

    def forward(self, input):
        T, B, C = input.shape[0], input.shape[1], input.shape[2]
        Cg = C // self.group_num

        xc = input.contiguous()
        # moment2 logical shape (T, B, C) contiguous strides
        m2_shape = (T, B, C)
        m2_str = (B * C, C, 1)

        out_shape = torch.broadcast_shapes(xc.shape, m2_shape)
        ndim = len(out_shape)
        assert ndim <= 4
        s = [1] * (4 - ndim) + list(out_shape)

        a_shape, a_str = _pad(xc.shape, xc.stride(), 4)
        b_shape, b_str = _pad(m2_shape, m2_str, 4)
        for i in range(4):
            if a_shape[i] == 1:
                a_str[i] = 0
            if b_shape[i] == 1:
                b_str[i] = 0

        out = torch.empty(out_shape, device=input.device, dtype=input.dtype)
        n = out.numel()
        BLOCK = 1024
        CG = triton.next_power_of_2(Cg)
        grid = (triton.cdiv(n, BLOCK),)
        _fused_kernel[grid](xc, out, n,
                            s[0], s[1], s[2], s[3],
                            a_str[0], a_str[1], a_str[2], a_str[3],
                            b_str[0], b_str[1], b_str[2], b_str[3],
                            Cg, self.eps, BLOCK=BLOCK, CG=CG, num_warps=8)
        return out
