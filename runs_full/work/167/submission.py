import torch
import torch.nn as nn
import triton
import triton.language as tl


def _pool1d_matrix(in_size):
    out_size = (in_size + 2 - 3) // 2 + 1
    M = torch.zeros((out_size, in_size), dtype=torch.float32)
    for o in range(out_size):
        for k in range(3):
            idx = 2 * o - 1 + k
            if 0 <= idx < in_size:
                M[o, idx] = 1.0 / 3.0
    return M, out_size


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@triton.jit
def _fused_pool_kernel(x_ptr, wh_ptr, ww_ptr, out_ptr,
                       H, W, OH, OW,
                       MH: tl.constexpr, MW: tl.constexpr,
                       MOH: tl.constexpr, MOW: tl.constexpr):
    nc = tl.program_id(0)
    a = tl.arange(0, MH)
    b = tl.arange(0, MW)
    oi = tl.arange(0, MOH)
    oj = tl.arange(0, MOW)

    x_idx = (nc * H + a[:, None]) * W + b[None, :]
    x_mask = (a[:, None] < H) & (b[None, :] < W)
    x = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)

    wh = tl.load(wh_ptr + oi[:, None] * H + a[None, :],
                 mask=(oi[:, None] < OH) & (a[None, :] < H), other=0.0)
    ww = tl.load(ww_ptr + oj[:, None] * W + b[None, :],
                 mask=(oj[:, None] < OW) & (b[None, :] < W), other=0.0)

    tmp = tl.sum(wh[:, :, None] * x[None, :, :], axis=1)
    out = tl.sum(tmp[:, None, :] * ww[None, :, :], axis=2)

    o_idx = (nc * OH + oi[:, None]) * OW + oj[None, :]
    o_mask = (oi[:, None] < OH) & (oj[None, :] < OW)
    tl.store(out_ptr + o_idx, out, mask=o_mask)


class InputInjectionNew(nn.Module):
    def __init__(self, num_downsampling):
        super(InputInjectionNew, self).__init__()
        self.num_downsampling = num_downsampling
        self.pool = nn.ModuleList()
        for i in range(num_downsampling):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))
        self._cache = {}

    def _get(self, H, W, device):
        key = (H, W, device)
        c = self._cache.get(key)
        if c is not None:
            return c
        Wh = torch.eye(H, dtype=torch.float32)
        Ww = torch.eye(W, dtype=torch.float32)
        oh, ow = H, W
        for _ in range(self.num_downsampling):
            Ph, oh = _pool1d_matrix(oh)
            Pw, ow = _pool1d_matrix(ow)
            Wh = Ph @ Wh
            Ww = Pw @ Ww
        c = (Wh.contiguous().to(device), Ww.contiguous().to(device), oh, ow,
             _next_pow2(H), _next_pow2(W), _next_pow2(oh), _next_pow2(ow))
        self._cache[key] = c
        return c

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.contiguous()
        Wh, Ww, OH, OW, MH, MW, MOH, MOW = self._get(H, W, x.device)
        out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
        grid = (N * C,)
        _fused_pool_kernel[grid](x, Wh, Ww, out, H, W, OH, OW,
                                 MH=MH, MW=MW, MOH=MOH, MOW=MOW, num_warps=1)
        return out
