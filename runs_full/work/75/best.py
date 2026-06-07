import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _spp_fused(x_ptr, out_ptr, sz_ptr, oh_ptr, ow_ptr, base_ptr,
               n_C, C, H, W, cells_per_c, total,
               MAX_K: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    nc = offs // cells_per_c
    cell = offs % cells_per_c

    sz = tl.load(sz_ptr + cell, mask=mask, other=1)
    oh = tl.load(oh_ptr + cell, mask=mask, other=0)
    ow = tl.load(ow_ptr + cell, mask=mask, other=0)
    cbase = tl.load(base_ptr + cell, mask=mask, other=0)

    h_start = (oh * H) // sz
    h_end = ((oh + 1) * H + sz - 1) // sz
    w_start = (ow * W) // sz
    w_end = ((ow + 1) * W + sz - 1) // sz

    base = nc * H * W
    acc = tl.full((BLOCK,), -float('inf'), tl.float32)
    for i in tl.static_range(MAX_K):
        for j in tl.static_range(MAX_K):
            h = h_start + i
            w = w_start + j
            valid = mask & (h < h_end) & (w < w_end)
            ptr = base + h * W + w
            v = tl.load(x_ptr + ptr, mask=valid, other=-float('inf'))
            acc = tl.maximum(acc, v)

    n_idx = nc // C
    c = nc % C
    out_idx = n_idx * (C * cells_per_c) + c * cells_per_c + cbase + (oh * sz + ow)
    tl.store(out_ptr + out_idx, acc, mask=mask)


class SppPoolingNew(nn.Module):

    def __init__(self, levels=[1, 2, 4]):
        super(SppPoolingNew, self).__init__()
        self.Pools = nn.ModuleList([nn.AdaptiveMaxPool2d((i, i)) for i in
            levels])
        self.levels = list(levels)
        sz_l, oh_l, ow_l, base_l = [], [], [], []
        cb = 0
        for sz in self.levels:
            for o in range(sz):
                for w in range(sz):
                    sz_l.append(sz); oh_l.append(o); ow_l.append(w); base_l.append(cb)
            cb += sz * sz
        self._sz = torch.tensor(sz_l, dtype=torch.int32)
        self._oh = torch.tensor(oh_l, dtype=torch.int32)
        self._ow = torch.tensor(ow_l, dtype=torch.int32)
        self._base = torch.tensor(base_l, dtype=torch.int32)
        self._cells = cb

    def forward(self, x):
        assert len(x.size()) == 4, '输入形状不满足(n,c,w,w)'
        n, C, H, W = x.shape
        x = x.contiguous()
        cells_per_c = self._cells
        if self._sz.device != x.device:
            self._sz = self._sz.to(x.device)
            self._oh = self._oh.to(x.device)
            self._ow = self._ow.to(x.device)
            self._base = self._base.to(x.device)
        out = torch.empty((n, C * cells_per_c), device=x.device, dtype=x.dtype)
        n_C = n * C
        total = n_C * cells_per_c
        max_sz = min(self.levels) if False else max(self.levels)
        min_sz = min(self.levels)
        MAX_K = (H + min_sz - 1) // min_sz + 1
        BLOCK = 256
        grid = (triton.cdiv(total, BLOCK),)
        _spp_fused[grid](x, out, self._sz, self._oh, self._ow, self._base,
                         n_C, C, H, W, cells_per_c, total,
                         MAX_K=MAX_K, BLOCK=BLOCK, num_warps=4)
        return out
