import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _minpool3d_kernel(
    x_ptr, out_ptr,
    C, D, H, W,
    OD, OH, OW,
    SD, SH, SW,
    total,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK: tl.constexpr, ONEBLOCK: tl.constexpr,
):
    if ONEBLOCK:
        offs = tl.arange(0, BLOCK)
    else:
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    ow = offs % OW
    t = offs // OW
    oh = t % OH
    t = t // OH
    od = t % OD
    t = t // OD
    c = t % C
    n = t // C

    d0 = od * SD
    h0 = oh * SH
    w0 = ow * SW
    base = ((n * C + c) * D) * H * W

    acc = tl.full((BLOCK,), float('inf'), tl.float32)
    for kd in range(0, KD):
        d = d0 + kd
        for kh in range(0, KH):
            h = h0 + kh
            for kw in range(0, KW):
                w = w0 + kw
                idx = base + (d * H + h) * W + w
                val = tl.load(x_ptr + idx, mask=mask, other=float('inf'))
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
        st = stride if stride is not None else kernel_size
        self.K = _triple(kernel_size)
        self.S = _triple(st)
        self.P = _triple(padding)
        self.Dl = _triple(dilation)
        self.ceil_mode = ceil_mode
        self._simple = (ndim == 3 and self.P == (0, 0, 0)
                        and self.Dl == (1, 1, 1) and not ceil_mode)

    def forward(self, x):
        if not self._simple:
            x_max = x.max()
            x = self.pool(x_max - x)
            return x_max - x

        orig_dim = x.dim()
        if orig_dim == 4:
            N, C, D, H, W = 1, *x.shape
        else:
            N, C, D, H, W = x.shape

        KD, KH, KW = self.K
        SD, SH, SW = self.S
        OD = (D - KD) // SD + 1
        OH = (H - KH) // SH + 1
        OW = (W - KW) // SW + 1

        out_shape = (C, OD, OH, OW) if orig_dim == 4 else (N, C, OD, OH, OW)
        out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        total = N * C * OD * OH * OW
        if total <= 1024:
            BLOCK = triton.next_power_of_2(total)
            _minpool3d_kernel[(1,)](
                x, out, C, D, H, W, OD, OH, OW, SD, SH, SW, total,
                KD=KD, KH=KH, KW=KW, BLOCK=BLOCK, ONEBLOCK=True, num_warps=1)
        else:
            BLOCK = 256
            grid = (triton.cdiv(total, BLOCK),)
            _minpool3d_kernel[grid](
                x, out, C, D, H, W, OD, OH, OW, SD, SH, SW, total,
                KD=KD, KH=KH, KW=KW, BLOCK=BLOCK, ONEBLOCK=False, num_warps=4)
        return out
