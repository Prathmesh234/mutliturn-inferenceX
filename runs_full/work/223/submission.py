import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _ssp_kernel(x_ptr, out_ptr, shift_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    shift = tl.load(shift_ptr)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    tl.store(out_ptr + offs, sp - shift, mask=mask)

class Shifted_softplusNew(nn.Module):
    def __init__(self):
        super(Shifted_softplusNew, self).__init__()
        self.act = nn.Softplus()
        self.shift = nn.Parameter(torch.tensor([0.6931]), False)

    def forward(self, X):
        out = torch.empty_like(X)
        n = X.numel()
        BLOCK = 256
        grid = (triton.cdiv(n, BLOCK),)
        _ssp_kernel[grid](X, out, self.shift, n, BLOCK=BLOCK, num_warps=2)
        return out
