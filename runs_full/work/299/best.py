import torch
from torch import nn
import triton
import triton.language as tl


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


@triton.jit
def _gelu(x):
    z = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    t = 2.0 / (1.0 + tl.exp(-2.0 * z)) - 1.0
    return 0.5 * x * (1.0 + t)


@triton.jit
def _block_kernel(x_ptr,
                  ln1w_ptr, ln1b_ptr, qkvw_ptr, qkvb_ptr,
                  projw_ptr, projb_ptr,
                  ln2w_ptr, ln2b_ptr,
                  fc1w_ptr, fc1b_ptr, fc2w_ptr, fc2b_ptr,
                  out_ptr,
                  N, H, HD, C, Hd, scale, eps,
                  stride_xm, stride_om,
                  stride_wq, stride_wp, stride_w1, stride_w2,
                  HAS_QKVB: tl.constexpr,
                  BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr):
    b = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_c = tl.arange(0, BLOCK_C)
    offs_h = tl.arange(0, BLOCK_H)
    mn = offs_n < N
    mc = offs_c < C
    mhh = offs_h < Hd
    rows = b * N + offs_n

    # ----- ln1 -----
    x = tl.load(x_ptr + rows[:, None] * stride_xm + offs_c[None, :],
                mask=mn[:, None] & mc[None, :], other=0.0)
    mean = tl.sum(x, axis=1) / C
    xc = tl.where(mc[None, :], x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    l1w = tl.load(ln1w_ptr + offs_c, mask=mc, other=0.0)
    l1b = tl.load(ln1b_ptr + offs_c, mask=mc, other=0.0)
    xn = xc * rstd[:, None] * l1w[None, :] + l1b[None, :]

    # ----- qkv (full) -----
    wq = tl.load(qkvw_ptr + offs_c[:, None] * stride_wq + offs_c[None, :],
                 mask=mc[:, None] & mc[None, :], other=0.0)
    wk = tl.load(qkvw_ptr + (C + offs_c)[:, None] * stride_wq + offs_c[None, :],
                 mask=mc[:, None] & mc[None, :], other=0.0)
    wv = tl.load(qkvw_ptr + (2 * C + offs_c)[:, None] * stride_wq + offs_c[None, :],
                 mask=mc[:, None] & mc[None, :], other=0.0)
    Q = tl.dot(xn, tl.trans(wq), input_precision="ieee")
    K = tl.dot(xn, tl.trans(wk), input_precision="ieee")
    V = tl.dot(xn, tl.trans(wv), input_precision="ieee")
    if HAS_QKVB:
        bq = tl.load(qkvb_ptr + offs_c, mask=mc, other=0.0)
        bk = tl.load(qkvb_ptr + C + offs_c, mask=mc, other=0.0)
        bv = tl.load(qkvb_ptr + 2 * C + offs_c, mask=mc, other=0.0)
        Q = Q + bq[None, :]
        K = K + bk[None, :]
        V = V + bv[None, :]

    # ----- multi-head attention via contraction masking -----
    col_mask = offs_n[None, :] < N
    ctx = tl.zeros((BLOCK_N, BLOCK_C), tl.float32)
    for h in range(H):
        hmask = ((offs_c >= h * HD) & (offs_c < h * HD + HD))[None, :]
        Qm = tl.where(hmask, Q, 0.0)
        scores = tl.dot(Qm, tl.trans(K), input_precision="ieee") * scale
        scores = tl.where(col_mask, scores, -float('inf'))
        m = tl.max(scores, axis=1)
        e = tl.exp(scores - m[:, None])
        e = tl.where(col_mask, e, 0.0)
        p = e / tl.sum(e, axis=1)[:, None]
        Vm = tl.where(hmask, V, 0.0)
        ctx += tl.dot(p, Vm, input_precision="ieee")

    # ----- proj + residual -----
    wp = tl.load(projw_ptr + offs_c[:, None] * stride_wp + offs_c[None, :],
                 mask=mc[:, None] & mc[None, :], other=0.0)
    x1 = tl.dot(ctx, tl.trans(wp), input_precision="ieee")
    bp = tl.load(projb_ptr + offs_c, mask=mc, other=0.0)
    x1 = x1 + bp[None, :] + x

    # ----- ln2 -----
    x1m = tl.where(mc[None, :], x1, 0.0)
    mean2 = tl.sum(x1m, axis=1) / C
    xc2 = tl.where(mc[None, :], x1 - mean2[:, None], 0.0)
    var2 = tl.sum(xc2 * xc2, axis=1) / C
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    l2w = tl.load(ln2w_ptr + offs_c, mask=mc, other=0.0)
    l2b = tl.load(ln2b_ptr + offs_c, mask=mc, other=0.0)
    xn2 = xc2 * rstd2[:, None] * l2w[None, :] + l2b[None, :]

    # ----- mlp -----
    w1 = tl.load(fc1w_ptr + offs_h[:, None] * stride_w1 + offs_c[None, :],
                 mask=mhh[:, None] & mc[None, :], other=0.0)
    h1 = tl.dot(xn2, tl.trans(w1), input_precision="ieee")
    b1 = tl.load(fc1b_ptr + offs_h, mask=mhh, other=0.0)
    h1 = _gelu(h1 + b1[None, :])
    h1 = tl.where(mhh[None, :], h1, 0.0)
    w2 = tl.load(fc2w_ptr + offs_c[:, None] * stride_w2 + offs_h[None, :],
                 mask=mc[:, None] & mhh[None, :], other=0.0)
    o = tl.dot(h1, tl.trans(w2), input_precision="ieee")
    b2 = tl.load(fc2b_ptr + offs_c, mask=mc, other=0.0)
    o = o + b2[None, :] + x1

    tl.store(out_ptr + rows[:, None] * stride_om + offs_c[None, :],
             o, mask=mn[:, None] & mc[None, :])


class _Attn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)


class BlockNew(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _Attn(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, N, C = x.shape
        M = B * N
        xf = x.reshape(M, C).contiguous()
        H = self.attn.num_heads
        HD = C // H
        Hd = self.mlp.fc1.weight.shape[0]
        BLOCK_N = max(16, triton.next_power_of_2(N))
        BLOCK_C = max(16, triton.next_power_of_2(C))
        BLOCK_H = max(16, triton.next_power_of_2(Hd))
        out = torch.empty((M, C), device=x.device, dtype=x.dtype)
        qkvb = self.attn.qkv.bias
        _block_kernel[(B,)](
            xf,
            self.norm1.weight, self.norm1.bias,
            self.attn.qkv.weight, qkvb if qkvb is not None else xf,
            self.attn.proj.weight, self.attn.proj.bias,
            self.norm2.weight, self.norm2.bias,
            self.mlp.fc1.weight, self.mlp.fc1.bias,
            self.mlp.fc2.weight, self.mlp.fc2.bias,
            out,
            N, H, HD, C, Hd, self.attn.scale, self.norm1.eps,
            xf.stride(0), out.stride(0),
            self.attn.qkv.weight.stride(0), self.attn.proj.weight.stride(0),
            self.mlp.fc1.weight.stride(0), self.mlp.fc2.weight.stride(0),
            HAS_QKVB=qkvb is not None,
            BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C, BLOCK_H=BLOCK_H, num_warps=1)
        return out.reshape(B, N, C)
