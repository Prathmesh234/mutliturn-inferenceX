import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _plu_kernel(x_ptr, out_ptr, n_elements, alpha, b1, b2, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    ax = alpha * x
    out = tl.maximum(ax + b1, tl.minimum(ax + b2, x))
    tl.store(out_ptr + offs, out, mask=mask)


class PLUNew(nn.Module):
    def __init__(self, alpha=0.1, c=1):
        super().__init__()
        self.alpha = alpha
        self.c = c

    def forward(self, x):
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        b1 = self.alpha * self.c - self.c
        b2 = self.c - self.alpha * self.c
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _plu_kernel[grid](x, out, n, self.alpha, b1, b2,
                          BLOCK_SIZE=BLOCK_SIZE, num_warps=4, num_stages=2)
        return out

    def __repr__(self):
        s = '{name} ({alhpa}, {c})'
        return s.format(name=self.__class__.__name__, **self.__dict__)
