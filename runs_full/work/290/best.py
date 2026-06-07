import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr, w_ptr, out_ptr, NHW, HW, H, W, C: tl.constexpr,
                  K: tl.constexpr, PAD: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NHW
    n = offs // HW
    hw = offs % HW
    h = hw // W
    w = hw % W
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for kh in range(K):
        ih = h + kh - PAD
        for kw in range(K):
            iw = w + kw - PAD
            valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            base = n * C * HW + ih * W + iw
            s = tl.zeros((BLOCK,), dtype=tl.float32)
            m = tl.full((BLOCK,), -float('inf'), dtype=tl.float32)
            for c in range(C):
                v = tl.load(x_ptr + base + c * HW, mask=valid, other=0.0).to(tl.float32)
                s += v
                m = tl.maximum(m, v)
            avg = s / C
            w0 = tl.load(w_ptr + kh * K + kw)
            w1 = tl.load(w_ptr + (K + kh) * K + kw)
            contrib = avg * w0 + m * w1
            acc += tl.where(valid, contrib, 0.0)
    out = 1.0 / (1.0 + tl.exp(-acc))
    tl.store(out_ptr + offs, out, mask=mask)


class SpatialAttentionNew(nn.Module):

    def __init__(self, kernel=3):
        super(SpatialAttentionNew, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.kernel = kernel

    def forward(self, x):
        x = x.contiguous()
        N, C, H, W = x.shape
        HW = H * W
        NHW = N * HW
        out = torch.empty((N, 1, H, W), device=x.device, dtype=torch.float32)
        BLOCK = 256
        grid = (triton.cdiv(NHW, BLOCK),)
        K = self.kernel
        PAD = K // 2
        w = self.conv1.weight.contiguous()
        _fused_kernel[grid](x, w, out, NHW, HW, H, W, C=C, K=K, PAD=PAD,
                            BLOCK=BLOCK, num_warps=4)
        return out.to(x.dtype)
