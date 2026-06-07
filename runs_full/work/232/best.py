import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gem_kernel(x_ptr, out_ptr, rows, n_spatial, p, inv_p, eps,
                BLOCK_R: tl.constexpr, BLOCK_S: tl.constexpr):
    pid = tl.program_id(0)
    row = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    col = tl.arange(0, BLOCK_S)
    rmask = row < rows
    cmask = col < n_spatial
    ptr = x_ptr + row[:, None] * n_spatial + col[None, :]
    mask = rmask[:, None] & cmask[None, :]
    x = tl.load(ptr, mask=mask, other=0.0)
    x = tl.maximum(x, eps)
    x = tl.exp(p * tl.log(x))
    s = tl.sum(x, axis=1)
    mean = s / n_spatial
    res = tl.exp(inv_p * tl.log(mean))
    tl.store(out_ptr + row, res, mask=rmask)


class GeneralizedMeanPoolingNew(nn.Module):
    def __init__(self, norm=3, output_size=(1, 1), eps=1e-06, *args, **kwargs):
        super().__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        assert self.output_size == (1, 1) or self.output_size == 1
        N, C, H, W = x.shape
        x = x.contiguous()
        rows = N * C
        n_spatial = H * W
        out = torch.empty((rows,), device=x.device, dtype=x.dtype)
        BLOCK_S = triton.next_power_of_2(n_spatial)
        BLOCK_R = triton.next_power_of_2(rows)
        grid = (1,)
        _gem_kernel[grid](x, out, rows, n_spatial, self.p, 1.0 / self.p, self.eps,
                          BLOCK_R=BLOCK_R, BLOCK_S=BLOCK_S, num_warps=1)
        return out.view(N, C, 1, 1)
