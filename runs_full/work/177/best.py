import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ppm_kernel(feats_ptr, out_ptr, meta_ptr, H, W, total_cols,
                BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    pid_nc = tl.program_id(0)
    pid_o = tl.program_id(1)
    hstart = tl.load(meta_ptr + pid_o * 4 + 0)
    hend = tl.load(meta_ptr + pid_o * 4 + 1)
    wstart = tl.load(meta_ptr + pid_o * 4 + 2)
    wend = tl.load(meta_ptr + pid_o * 4 + 3)
    rh = tl.arange(0, BLOCK_H)
    rw = tl.arange(0, BLOCK_W)
    hmask = (rh >= hstart) & (rh < hend)
    wmask = (rw >= wstart) & (rw < wend)
    base = pid_nc * H * W
    offs = base + rh[:, None] * W + rw[None, :]
    mask = hmask[:, None] & wmask[None, :]
    vals = tl.load(feats_ptr + offs, mask=mask, other=0.0)
    s = tl.sum(vals)
    cnt = (hend - hstart) * (wend - wstart)
    tl.store(out_ptr + pid_nc * total_cols + pid_o, s / cnt)


class PPMConcatNew(nn.ModuleList):
    def __init__(self, pool_scales=(1, 3, 6, 8)):
        super(PPMConcatNew, self).__init__(
            [nn.AdaptiveAvgPool2d(pool_scale) for pool_scale in pool_scales])
        self.pool_scales = pool_scales
        self._meta_cache = {}

    def _get_meta(self, H, W, device):
        key = (H, W, device)
        m = self._meta_cache.get(key)
        if m is None:
            rows = []
            for s in self.pool_scales:
                for oi in range(s):
                    hs = (oi * H) // s
                    he = ((oi + 1) * H + s - 1) // s
                    for oj in range(s):
                        ws = (oj * W) // s
                        we = ((oj + 1) * W + s - 1) // s
                        rows.append([hs, he, ws, we])
            m = torch.tensor(rows, dtype=torch.int32, device=device).contiguous()
            self._meta_cache[key] = m
        return m

    def forward(self, feats):
        feats = feats.contiguous()
        N, C, H, W = feats.shape
        total_cols = sum(s * s for s in self.pool_scales)
        out = torch.empty((N, C, total_cols), device=feats.device, dtype=feats.dtype)
        meta = self._get_meta(H, W, feats.device)
        BLOCK_H = triton.next_power_of_2(H)
        BLOCK_W = triton.next_power_of_2(W)
        grid = (N * C, total_cols)
        _ppm_kernel[grid](feats, out, meta, H, W, total_cols,
                          BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, num_warps=1)
        return out
