import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _ln_kernel(x_ptr, out_ptr, gamma_ptr, beta_ptr, M, C, HW, eps,
               AFFINE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < M
    x = tl.load(x_ptr + row * M + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / M
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / (M - 1)
    std = tl.sqrt(var)
    inv = 1.0 / (std + eps)
    y = xc * inv
    if AFFINE:
        c = offs // HW
        g = tl.load(gamma_ptr + c, mask=mask, other=0.0)
        b = tl.load(beta_ptr + c, mask=mask, other=0.0)
        y = y * g + b
    tl.store(out_ptr + row * M + offs, y, mask=mask)


class LayerNormNew(nn.Module):

    def __init__(self, num_features, eps=1e-05, affine=True):
        super(LayerNormNew, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = x.contiguous()
        N = x.size(0)
        M = x.numel() // N
        C = self.num_features
        HW = M // C
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(M)
        if self.affine:
            g, b = self.gamma, self.beta
        else:
            g = b = x  # dummy
        _ln_kernel[(N,)](x, out, g, b, M, C, HW, self.eps,
                         AFFINE=self.affine, BLOCK=BLOCK,
                         num_warps=4)
        return out
