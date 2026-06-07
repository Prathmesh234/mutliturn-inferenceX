import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _mul_kernel(x_ptr, out_ptr, n_elements, beta, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * beta, mask=mask)


class MomentumNetSideNew(nn.Module):
    def __init__(self, beta: float):
        super(MomentumNetSideNew, self).__init__()
        self.beta = beta

    def forward(self, inp: torch.Tensor):
        inp = inp.contiguous()
        out = torch.empty_like(inp)
        n = inp.numel()
        BLOCK_SIZE = triton.next_power_of_2(n)
        _mul_kernel[(1,)](inp, out, n, self.beta, BLOCK_SIZE=BLOCK_SIZE, num_warps=2)
        return out
