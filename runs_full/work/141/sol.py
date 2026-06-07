import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _minpool3d_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    DD, DH, DW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * OD * OH * OW
    mask = offs < total

    ow = offs % OW
    t = offs // OW
    oh = t % OH
    t = t // OH
    od = t % OD
    t = t // OD
    c = t % C
    n = t // C

    d0 = od * SD - PD
    h0 = oh * SH - PH
    w0 = ow * SW - PW

    base = ((n * C + c) * D) * H * W

    acc = tl.full((BLOCK,), float('inf'), tl.float32)
    for kd in range(0, KD):
        d = d0 + kd * DD
        vd = (d >= 0) & (d < D)
        for kh in range(0, KH):
            h = h0 + kh * DH
            vh = (h >= 0) & (h < H)
            for kw in range(0, KW):
                w = w0 + kw * DW
                vw = (w >= 0) & (w < W)
                v = vd & vh & vw & mask
                idx = base + (d * H + h) * W + w
                val = tl.load(x_ptr + idx, mask=v, other=float('inf'))
                acc = tl.minimum(acc, val)

    tl.store(out_ptr + offs, acc, mask=mask)


def _triple(x):
    if isinstance(x, (tuple, list)):
        return int(x[0]), int(x[1]), int(x[2])
    return int(x), int(x), int(x)


class MinPoolNew(nn.Module):
    def __init__(self, kernel_size, ndim=3, stride=None, padding=0,
                 dilation=1, return_indices=False, ceil_mode=False):
        super(MinPoolNew, self).__init__()
        self.pool = getattr(nn, f'MaxPool{ndim}d')(kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            return_indices=return_indices, ceil_mode=ceil_mode)
        self.ndim = ndim
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, x):
        if self.ndim != 3:
            x_max = x.max()
            x = self.pool(x_max - x)
            return x_max - x

        orig_dim = x.dim()
        if orig_dim == 4:
            x = x.unsqueeze(0)
        x = x.contiguous()
        N, C, D, H, W = x.shape

        KD, KH, KW = _triple(self.kernel_size)
        SD, SH, SW = _triple(self.stride)
        PD, PH, PW = _triple(self.padding)
        DD, DH, DW = _triple(self.dilation)

        def out_dim(L, k, s, p, d):
            num = L + 2 * p - d * (k - 1) - 1
            if self.ceil_mode:
                o = -(-num // s) + 1
            else:
                o = num // s + 1
            return o

        OD = out_dim(D, KD, SD, PD, DD)
        OH = out_dim(H, KH, SH, PH, DH)
        OW = out_dim(W, KW, SW, PW, DW)

        out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)
        total = N * C * OD * OH * OW
        BLOCK = 256
        grid = (triton.cdiv(total, BLOCK),)
        _minpool3d_kernel[grid](
            x, out,
            N, C, D, H, W,
            OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            DD, DH, DW,
            BLOCK=BLOCK, num_warps=4,
        )
        if orig_dim == 4:
            out = out.squeeze(0)
        return out
