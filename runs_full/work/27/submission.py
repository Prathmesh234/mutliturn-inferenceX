import torch
import torch.nn as nn
import torch.nn.init as init
import triton
import triton.language as tl


@triton.jit
def _l2norm_kernel(x_ptr, w_ptr, out_ptr, n_spatial, HW, C: tl.constexpr,
                   eps, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_spatial
    n = offs // HW
    hw = offs % HW
    base = n * (C * HW) + hw
    cidx = tl.arange(0, C)
    addr = base[:, None] + cidx[None, :] * HW
    m2 = mask[:, None]
    x = tl.load(x_ptr + addr, mask=m2, other=0.0).to(tl.float32)
    sumsq = tl.sum(x * x, axis=1)
    inv = 1.0 / (tl.sqrt(sumsq) + eps)
    w = tl.load(w_ptr + cidx).to(tl.float32)
    out = w[None, :] * x * inv[:, None]
    tl.store(out_ptr + addr, out, mask=m2)


class L2NormNew(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2NormNew, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        N, C, H, W = x.shape
        HW = H * W
        n_spatial = N * HW
        out = torch.empty_like(x)
        BLOCK = 1024
        grid = (triton.cdiv(n_spatial, BLOCK),)
        _l2norm_kernel[grid](x, self.weight, out, n_spatial, HW, C,
                             self.eps, BLOCK=BLOCK, num_warps=2)
        return out
