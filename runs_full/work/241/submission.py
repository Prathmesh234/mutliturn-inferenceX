import torch
import torch.nn as nn
import triton
import triton.language as tl


def _np2(n):
    return 1 << (max(int(n) - 1, 0)).bit_length()


@triton.jit
def _fused_kernel(x_ptr, iw_ptr, ib_ptr, ow_ptr, ob_ptr, o_ptr,
                  L, E, Dh, scale, H: tl.constexpr,
                  BL: tl.constexpr, BD: tl.constexpr, BE: tl.constexpr):
    offs_l = tl.arange(0, BL)
    offs_e = tl.arange(0, BE)
    offs_d = tl.arange(0, BD)
    offs_n = tl.arange(0, BE)
    xm = (offs_l[:, None] < L) & (offs_e[None, :] < E)
    x = tl.load(x_ptr + offs_l[:, None] * E + offs_e[None, :], mask=xm, other=0.0)
    dm = offs_d < Dh
    wm = (offs_d[:, None] < Dh) & (offs_e[None, :] < E)
    acc = tl.zeros((BL, BE), dtype=tl.float32)
    for h in range(H):
        row = h * Dh + offs_d
        wq = tl.load(iw_ptr + row[:, None] * E + offs_e[None, :], mask=wm, other=0.0)
        wk = tl.load(iw_ptr + (E + row)[:, None] * E + offs_e[None, :], mask=wm, other=0.0)
        wv = tl.load(iw_ptr + (2 * E + row)[:, None] * E + offs_e[None, :], mask=wm, other=0.0)
        bq = tl.load(ib_ptr + row, mask=dm, other=0.0)
        bk = tl.load(ib_ptr + E + row, mask=dm, other=0.0)
        bv = tl.load(ib_ptr + 2 * E + row, mask=dm, other=0.0)
        q = tl.sum(x[:, None, :] * wq[None, :, :], axis=2) + bq[None, :]
        k = tl.sum(x[:, None, :] * wk[None, :, :], axis=2) + bk[None, :]
        v = tl.sum(x[:, None, :] * wv[None, :, :], axis=2) + bv[None, :]
        scores = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale
        scores = tl.where(offs_l[None, :] < L, scores, -1e30)
        scores = scores - tl.max(scores, axis=1)[:, None]
        p = tl.exp(scores)
        p = p / tl.sum(p, axis=1)[:, None]
        ctx = tl.sum(p[:, :, None] * v[None, :, :], axis=1)
        col = h * Dh + offs_d
        owm = (offs_n[:, None] < E) & (offs_d[None, :] < Dh)
        owh = tl.load(ow_ptr + offs_n[:, None] * E + col[None, :], mask=owm, other=0.0)
        acc += tl.sum(ctx[:, None, :] * owh[None, :, :], axis=2)
    ob = tl.load(ob_ptr + offs_n, mask=offs_n < E, other=0.0)
    acc += ob[None, :]
    tl.store(o_ptr + offs_l[:, None] * E + offs_n[None, :], acc,
             mask=(offs_l[:, None] < L) & (offs_n[None, :] < E))


class SelfAttentionNew(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.fn = nn.MultiheadAttention(*args, **kwargs)

    def forward(self, x):
        E = self.fn.embed_dim
        H = self.fn.num_heads
        Dh = E // H
        x2 = x.contiguous()
        L = x2.shape[0]
        iw = self.fn.in_proj_weight
        ib = self.fn.in_proj_bias
        if ib is None:
            ib = torch.zeros(3 * E, device=x.device, dtype=x.dtype)
        ow = self.fn.out_proj.weight
        ob = self.fn.out_proj.bias
        if ob is None:
            ob = torch.zeros(E, device=x.device, dtype=x.dtype)
        out = torch.empty((L, E), device=x.device, dtype=x.dtype)
        scale = Dh ** -0.5
        _fused_kernel[(1,)](
            x2, iw, ib, ow, ob, out, L, E, Dh, scale, H,
            BL=_np2(L), BD=_np2(Dh), BE=_np2(E), num_warps=1)
        return out
