import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ncut_kernel(labels_ptr, weights_ptr, out_ptr,
                 B, K, Binv, H: tl.constexpr, W: tl.constexpr, L: tl.constexpr,
                 radius: tl.constexpr, R: tl.constexpr, RR: tl.constexpr,
                 BLOCK_RR: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // K
    k = pid % K

    offs = tl.arange(0, BLOCK_RR)
    mask_rr = offs < RR
    a = offs // R
    c = offs % R

    labels_k_base = (b * K + k) * L

    numerator = 0.0
    denom = 0.0
    for l in range(L):
        oy = l // W
        ox = l % W
        p_center = tl.load(labels_ptr + labels_k_base + l)
        iy = oy + a - radius
        ix = ox + c - radius
        valid = mask_rr & (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W)
        w = tl.load(weights_ptr + (b * L + l) * RR + offs, mask=mask_rr, other=0.0)
        prob = tl.load(labels_ptr + labels_k_base + iy * W + ix, mask=valid, other=0.0)
        numerator += p_center * tl.sum(w * prob)
        denom += p_center * tl.sum(w)

    tl.atomic_add(out_ptr, -(numerator / denom) * Binv)


class NCutLossOptimizedNew(nn.Module):
    def __init__(self, radius: int = 5):
        super().__init__()
        self.radius = radius

    def forward(self, labels, weights):
        labels = labels.contiguous()
        weights = weights.contiguous()
        B, K, H, W = labels.shape
        L = H * W
        radius = self.radius
        R = 2 * radius + 1
        RR = R * R
        BLOCK_RR = triton.next_power_of_2(RR)

        out = torch.full((), float(K), device=labels.device, dtype=torch.float32)
        _ncut_kernel[(B * K,)](
            labels, weights, out,
            B, K, 1.0 / B, H, W, L, radius, R, RR, BLOCK_RR,
            num_warps=2,
        )
        return out
