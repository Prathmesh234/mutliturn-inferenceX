import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _pool_kernel(d_ptr, mask_ptr, dres_ptr, mres_ptr,
                 N, C, H, W, OH, OW, STRIDE,
                 LARGE: tl.constexpr, BLOCK: tl.constexpr, KS: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * OH * OW
    valid = offs < total

    ow = offs % OW
    t = offs // OW
    oh = t % OH
    t = t // OH
    c = t % C
    n = t // C

    base_h = oh * STRIDE
    base_w = ow * STRIDE
    nc = (n * C + c) * H

    neg_inf = -3.4e38
    max_enc = tl.full((BLOCK,), neg_inf, tl.float32)
    max_mask = tl.full((BLOCK,), neg_inf, tl.float32)

    for kh in tl.static_range(KS):
        for kw in tl.static_range(KS):
            h = base_h + kh
            w = base_w + kw
            in_range = valid & (h < H) & (w < W)
            idx = (nc + h) * W + w
            d = tl.load(d_ptr + idx, mask=in_range, other=0.0)
            m = tl.load(mask_ptr + idx, mask=in_range, other=0.0)
            enc = -(1.0 - m) * LARGE - d
            enc = tl.where(in_range, enc, neg_inf)
            mm = tl.where(in_range, m, neg_inf)
            max_enc = tl.maximum(max_enc, enc)
            max_mask = tl.maximum(max_mask, mm)

    d_pooled = -max_enc
    d_result = d_pooled - (1.0 - max_mask) * LARGE
    tl.store(dres_ptr + offs, d_result, mask=valid)
    tl.store(mres_ptr + offs, max_mask, mask=valid)


class SparseDownSampleCloseNew(nn.Module):
    def __init__(self, stride):
        super(SparseDownSampleCloseNew, self).__init__()
        self.pooling = nn.MaxPool2d(stride, stride)
        self.large_number = 600
        self.stride = stride

    def forward(self, d, mask):
        d = d.contiguous()
        mask = mask.contiguous()
        N, C, H, W = d.shape
        s = self.stride
        OH = H // s
        OW = W // s
        dres = torch.empty((N, C, OH, OW), device=d.device, dtype=d.dtype)
        mres = torch.empty((N, C, OH, OW), device=d.device, dtype=d.dtype)
        total = N * C * OH * OW
        BLOCK = 256
        grid = (triton.cdiv(total, BLOCK),)
        _pool_kernel[grid](d, mask, dres, mres,
                           N, C, H, W, OH, OW, s,
                           LARGE=float(self.large_number), BLOCK=BLOCK, KS=s,
                           num_warps=2)
        return dres, mres
