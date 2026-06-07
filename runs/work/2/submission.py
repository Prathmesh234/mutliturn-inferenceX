import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _norm_affine_kernel(x_ptr, out_ptr, scale_ptr, bias_ptr,
                        n_cols, inner, last_size, outer_stride,
                        D1: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_cols
    o = offs // inner
    i = offs % inner
    last_idx = i % last_size
    base = o * outer_stride + i

    vals = []
    acc = tl.zeros([BLOCK], tl.float32)
    for j in range(D1):
        v = tl.load(x_ptr + base + j * inner, mask=mask, other=0.0).to(tl.float32)
        vals.append(v)
        acc += v * v
    inv = 1.0 / tl.sqrt(acc)

    s = tl.load(scale_ptr + last_idx, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + last_idx, mask=mask, other=0.0).to(tl.float32)

    for j in range(D1):
        out = vals[j] * inv * s + b
        tl.store(out_ptr + base + j * inner, out, mask=mask)


class CustomizeLayerNew(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.scale = nn.Parameter(torch.Tensor(self.in_dim))
        self.bias = nn.Parameter(torch.Tensor(self.in_dim))

    def forward(self, x):
        x = x.contiguous()
        out = torch.empty_like(x)
        D1 = x.shape[1]
        inner = 1
        for d in x.shape[2:]:
            inner *= d
        last_size = x.shape[-1]
        n_cols = x.shape[0] * inner
        outer_stride = D1 * inner
        BLOCK = 256
        grid = (triton.cdiv(n_cols, BLOCK),)
        _norm_affine_kernel[grid](x, out, self.scale, self.bias,
                                  n_cols, inner, last_size, outer_stride,
                                  D1=D1, BLOCK=BLOCK, num_warps=4)
        return out

    def __repr__(self):
        return 'CustomizedLayer(in_dim=%d)' % self.in_dim
