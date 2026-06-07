import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _spatial_attn_kernel(x_ptr, w_ptr, out_ptr, n_spatial, C, HW,
                         BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    s = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = s < n_spatial
    b = s // HW
    rem = s % HW
    base = b * (C * HW) + rem

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for c in range(C):
        xc = tl.load(x_ptr + base + c * HW, mask=mask, other=0.0)
        wc = tl.load(w_ptr + c)
        acc += xc * wc
    z = tl.sigmoid(acc)

    for c in range(C):
        xc = tl.load(x_ptr + base + c * HW, mask=mask, other=0.0)
        tl.store(out_ptr + base + c * HW, xc * z, mask=mask)


class SpatialAttention2dNew(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2dNew, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.contiguous()
        out = torch.empty_like(x)
        HW = H * W
        n_spatial = B * HW
        w = self.squeeze.weight.reshape(C).contiguous()
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_spatial, BLOCK_SIZE),)
        _spatial_attn_kernel[grid](x, w, out, n_spatial, C, HW,
                                   BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out
