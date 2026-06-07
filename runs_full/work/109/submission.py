import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _ln_kernel(x_ptr, w_ptr, b_ptr, out_ptr, M, N, eps, AFFINE: tl.constexpr,
               BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    xptr = x_ptr + row * N + cols
    x = tl.load(xptr, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    y = xc * rstd
    if AFFINE:
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bb = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = y * w + bb
    tl.store(out_ptr + row * N + cols, y, mask=mask)


class LayerNorm1dNew(nn.Module):
    def __init__(self, num_features, eps=1e-06, affine=True):
        super(LayerNorm1dNew, self).__init__()
        self.eps = eps
        self.num_features = num_features
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.fill_(0.0)

    def forward(self, inputs):
        b, t, n = inputs.shape
        x = inputs.contiguous()
        out = torch.empty_like(x)
        M = b * t
        BLOCK_N = triton.next_power_of_2(n)
        _ln_kernel[(M,)](x, self.weight, self.bias, out, M, n, self.eps,
                         self.affine, BLOCK_N=BLOCK_N, num_warps=1)
        return out
