import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rot180_kernel(in_ptr, out_ptr, n_elements, HW, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    n = offs // HW
    r = offs - n * HW
    src = n * HW + (HW - 1 - r)
    x = tl.load(in_ptr + src, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)


class Rot180New(nn.Module):
    def __init__(self) -> None:
        super(Rot180New, self).__init__()

    def forward(self, input: 'torch.Tensor') -> torch.Tensor:
        x = input.contiguous()
        HW = x.shape[-2] * x.shape[-1]
        n_elements = x.numel()
        out = torch.empty_like(x)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _rot180_kernel[grid](x, out, n_elements, HW, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
        return out

    def __repr__(self):
        return self.__class__.__name__
