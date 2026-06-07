import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _surface_loss_kernel(x_ptr, d_ptr, out_ptr, C, HW,
                         BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr):
    n = tl.program_id(0)
    base = n * C * HW
    offs_c = tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    acc = 0.0
    for s0 in range(0, HW, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        s_mask = offs_s < HW
        ptr = base + offs_c[:, None] * HW + offs_s[None, :]
        full_mask = c_mask[:, None] & s_mask[None, :]
        x = tl.load(x_ptr + ptr, mask=full_mask, other=-float('inf'))
        m = tl.max(x, axis=0)
        e = tl.exp(x - m[None, :])
        e = tl.where(full_mask, e, 0.0)
        denom = tl.sum(e, axis=0)
        sm = e / denom[None, :]
        d = tl.load(d_ptr + ptr, mask=full_mask, other=0.0)
        prod = tl.where(full_mask, sm * d, 0.0)
        acc += tl.sum(prod)
    out = acc / (C * HW)
    tl.store(out_ptr + n, out)


class SurfaceLossNew(nn.Module):

    def __init__(self, epsilon=1e-05, softmax=True):
        super(SurfaceLossNew, self).__init__()
        self.weight_map = []

    def forward(self, x, distmap):
        self.weight_map = distmap
        N, C = x.shape[0], x.shape[1]
        HW = x.numel() // (N * C)
        x = x.contiguous()
        distmap = distmap.contiguous()
        out = torch.empty(N, device=x.device, dtype=torch.float32)
        BLOCK_C = triton.next_power_of_2(C)
        BLOCK_S = 256
        _surface_loss_kernel[(N,)](x, distmap, out, C, HW,
                                   BLOCK_C=BLOCK_C, BLOCK_S=BLOCK_S,
                                   num_warps=4)
        return out.to(x.dtype)
