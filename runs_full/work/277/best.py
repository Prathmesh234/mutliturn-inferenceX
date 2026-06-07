import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _hflip_kernel(x_ptr, out_ptr, n, W, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    col = offs % W
    rev = offs - col + (W - 1 - col)
    tl.store(out_ptr + rev, x, mask=mask)


class HflipNew(nn.Module):
    def __init__(self) -> None:
        super(HflipNew, self).__init__()

    def forward(self, input: 'torch.Tensor') -> torch.Tensor:
        x = input.contiguous()
        W = x.shape[-1]
        n = x.numel()
        out = torch.empty_like(x)
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _hflip_kernel[grid](x, out, n, W, BLOCK=BLOCK, num_warps=4)
        return out

    def __repr__(self):
        return self.__class__.__name__
