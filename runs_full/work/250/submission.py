import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _l2norm_kernel(x_ptr, w_ptr, out_ptr, NS, S, C, eps,
                   BLOCK_SP: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SP + tl.arange(0, BLOCK_SP)
    mask = offs < NS
    n = offs // S
    s = offs % S
    base = n * (C * S) + s
    acc = tl.zeros((BLOCK_SP,), dtype=tl.float32)
    for c in range(0, C):
        ptr = base + c * S
        xv = tl.load(x_ptr + ptr, mask=mask, other=0.0).to(tl.float32)
        acc += xv * xv
    norm = tl.sqrt(acc) + eps
    for c in range(0, C):
        ptr = base + c * S
        xv = tl.load(x_ptr + ptr, mask=mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + c).to(tl.float32)
        res = wv * xv / norm
        tl.store(out_ptr + ptr, res, mask=mask)


class L2NormNew(nn.Module):
    def __init__(self, n_dims, scale=20.0, eps=1e-10):
        super(L2NormNew, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        N, C, H, W = x.shape
        S = H * W
        NS = N * S
        xc = x.contiguous()
        out = torch.empty_like(xc, dtype=torch.float32)
        w = self.weight.contiguous().float()
        BLOCK_SP = 64
        grid = (triton.cdiv(NS, BLOCK_SP),)
        _l2norm_kernel[grid](xc, w, out, NS, S, C, self.eps,
                             BLOCK_SP=BLOCK_SP, num_warps=1)
        return out.type_as(x)
