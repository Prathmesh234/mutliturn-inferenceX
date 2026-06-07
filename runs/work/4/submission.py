import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(x_ptr, a_ptr, b_ptr, out_ptr, N, eps,
                      BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / (N - 1)
    std = tl.sqrt(var)
    a = tl.load(a_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = a * xc / (std + eps) + b
    tl.store(out_ptr + row * N + cols, out, mask=mask)


class LayerNormNew(nn.Module):
    def __init__(self, features, eps=1e-06):
        super(LayerNormNew, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        N = x.shape[-1]
        x_c = x.contiguous()
        out = torch.empty_like(x_c)
        rows = x_c.numel() // N
        BLOCK_SIZE = triton.next_power_of_2(N)
        _layernorm_kernel[(rows,)](x_c.view(-1, N), self.a_2, self.b_2,
                                   out.view(-1, N), N, self.eps,
                                   BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
        return out.view_as(x)
