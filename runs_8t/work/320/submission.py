import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _block_kernel(
    inp_ptr, out_ptr,
    dw_w_ptr, dw_b_ptr,
    n_w_ptr, n_b_ptr,
    w1_ptr, b1_ptr,
    w2_ptr, b2_ptr,
    gamma_ptr,
    N, C, H, W, HW,
    eps,
    KH: tl.constexpr, KW: tl.constexpr, PAD: tl.constexpr,
    HAS_GAMMA: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_4C: tl.constexpr,
):
    t = tl.program_id(0)
    n = t // HW
    rem = t % HW
    h = rem // W
    w = rem % W

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = n * C * HW + offs_c * HW

    # depthwise conv
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for kh in range(KH):
        for kw in range(KW):
            ih = h + kh - PAD
            iw = w + kw - PAD
            valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            in_idx = base + ih * W + iw
            x = tl.load(inp_ptr + in_idx, mask=mask_c & valid, other=0.0).to(tl.float32)
            wt = tl.load(dw_w_ptr + offs_c * (KH * KW) + kh * KW + kw, mask=mask_c, other=0.0).to(tl.float32)
            acc += wt * x
    acc += tl.load(dw_b_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)

    # layernorm over C
    mean = tl.sum(acc, axis=0) / C
    xc = acc - mean
    var = tl.sum(xc * xc, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    nw = tl.load(n_w_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    nb = tl.load(n_b_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    xn = xc * rstd * nw + nb

    # linear1: hidden[j] = sum_c xn[c]*w1[j,c] + b1[j]
    offs_j = tl.arange(0, BLOCK_4C)
    mask_j = offs_j < (4 * C)
    w1 = tl.load(w1_ptr + offs_j[:, None] * C + offs_c[None, :],
                 mask=mask_j[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
    hidden = tl.sum(xn[None, :] * w1, axis=1)
    hidden += tl.load(b1_ptr + offs_j, mask=mask_j, other=0.0).to(tl.float32)
    # gelu (exact)
    hidden = 0.5 * hidden * (1.0 + tl.erf(hidden * 0.7071067811865476))

    # linear2: out[c] = sum_j hidden[j]*w2[c,j] + b2[c]
    w2 = tl.load(w2_ptr + offs_c[:, None] * (4 * C) + offs_j[None, :],
                 mask=mask_c[:, None] & mask_j[None, :], other=0.0).to(tl.float32)
    out = tl.sum(hidden[None, :] * w2, axis=1)
    out += tl.load(b2_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)

    if HAS_GAMMA:
        out = out * tl.load(gamma_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)

    # residual + store (NCHW)
    res = tl.load(inp_ptr + base + h * W + w, mask=mask_c, other=0.0).to(tl.float32)
    tl.store(out_ptr + base + h * W + w, res + out, mask=mask_c)



class _LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-06, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = normalized_shape,


class BlockNew(nn.Module):
    def __init__(self, dim, base_conv=nn.Conv2d, drop_path=0.0,
                 layer_scale_init_value=1e-06):
        super().__init__()
        self.dwconv = base_conv(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = _LayerNorm(dim, eps=1e-06)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()
        self.dim = dim

    def forward(self, x):
        x = x.contiguous()
        N, C, H, W = x.shape
        out = torch.empty_like(x)
        T = N * H * W
        BLOCK_C = triton.next_power_of_2(C)
        BLOCK_4C = triton.next_power_of_2(4 * C)
        gamma = self.gamma if self.gamma is not None else x
        _block_kernel[(T,)](
            x, out,
            self.dwconv.weight, self.dwconv.bias,
            self.norm.weight, self.norm.bias,
            self.pwconv1.weight, self.pwconv1.bias,
            self.pwconv2.weight, self.pwconv2.bias,
            gamma,
            N, C, H, W, H * W,
            1e-6,
            KH=7, KW=7, PAD=3,
            HAS_GAMMA=self.gamma is not None,
            BLOCK_C=BLOCK_C, BLOCK_4C=BLOCK_4C,
            num_warps=4,
        )
        return out
