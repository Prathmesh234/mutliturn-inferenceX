import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ln_kernel(x_ptr, w_ptr, b_ptr, out_ptr, n_rows, N, eps,
               HAS_W: tl.constexpr, HAS_B: tl.constexpr,
               BLOCK_N: tl.constexpr, ROWS: tl.constexpr):
    row_off = tl.arange(0, ROWS)
    col_off = tl.arange(0, BLOCK_N)
    col_mask = col_off < N
    row_mask = row_off < n_rows
    mask = row_mask[:, None] & col_mask[None, :]
    ptrs = x_ptr + row_off[:, None] * N + col_off[None, :]
    x = tl.load(ptrs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=1) / N
    xc = tl.where(col_mask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    y = xc * rstd[:, None]
    if HAS_W:
        w = tl.load(w_ptr + col_off, mask=col_mask, other=0.0)
        y = y * w[None, :]
    if HAS_B:
        b = tl.load(b_ptr + col_off, mask=col_mask, other=0.0)
        y = y + b[None, :]
    tl.store(out_ptr + row_off[:, None] * N + col_off[None, :], y, mask=mask)


class Fp32LayerNormNew(nn.LayerNorm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        N = 1
        for s in self.normalized_shape:
            N *= s
        self._N = N
        self._BLOCK_N = triton.next_power_of_2(N)
        self._cache = (0, 0)

    def forward(self, input):
        x = input.contiguous()
        N = self._N
        n_rows = x.numel() // N
        cr, cR = self._cache
        if cr == n_rows:
            ROWS = cR
        else:
            ROWS = triton.next_power_of_2(n_rows)
            self._cache = (n_rows, ROWS)
        out = torch.empty_like(x)
        w = self.weight
        b = self.bias
        _ln_kernel[(1,)](
            x, w if w is not None else x, b if b is not None else x,
            out, n_rows, N, self.eps,
            HAS_W=w is not None, HAS_B=b is not None,
            BLOCK_N=self._BLOCK_N, ROWS=ROWS, num_warps=1,
        )
        return out
