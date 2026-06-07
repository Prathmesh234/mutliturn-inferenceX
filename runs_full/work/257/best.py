import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _se_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
               C, HW, M,
               BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_HW: tl.constexpr):
    n = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    offs_m = tl.arange(0, BLOCK_M)
    offs_hw = tl.arange(0, BLOCK_HW)
    mask_c = offs_c < C
    mask_m = offs_m < M
    mask_hw = offs_hw < HW

    xp = x_ptr + n * C * HW + offs_c[:, None] * HW + offs_hw[None, :]
    xmask = mask_c[:, None] & mask_hw[None, :]
    xb = tl.load(xp, mask=xmask, other=0.0)
    pooled = tl.sum(xb, axis=1) / HW

    w1 = tl.load(w1_ptr + offs_m[:, None] * C + offs_c[None, :],
                 mask=mask_m[:, None] & mask_c[None, :], other=0.0)
    h = tl.sum(w1 * pooled[None, :], axis=1) + tl.load(b1_ptr + offs_m, mask=mask_m, other=0.0)
    h = tl.maximum(h, 0.0)

    w2 = tl.load(w2_ptr + offs_c[:, None] * M + offs_m[None, :],
                 mask=mask_c[:, None] & mask_m[None, :], other=0.0)
    s = tl.sum(w2 * h[None, :], axis=1) + tl.load(b2_ptr + offs_c, mask=mask_c, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-s))

    tl.store(out_ptr + n * C * HW + offs_c[:, None] * HW + offs_hw[None, :],
             xb * s[:, None], mask=xmask)


class SELayerNew(nn.Module):

    def __init__(self, in_channels, reduction):
        super(SELayerNew, self).__init__()
        mid_channels = in_channels // reduction
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, in_channels)

    def forward(self, x):
        n_batches, n_channels, H, W = x.size()
        x = x.contiguous()
        HW = H * W
        mid = self.fc1.weight.shape[0]
        out = torch.empty_like(x)
        _se_kernel[(n_batches,)](
            x, self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias, out,
            n_channels, HW, mid,
            BLOCK_C=triton.next_power_of_2(n_channels),
            BLOCK_M=triton.next_power_of_2(mid),
            BLOCK_HW=triton.next_power_of_2(HW),
            num_warps=4)
        return out
