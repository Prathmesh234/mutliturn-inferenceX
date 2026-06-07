import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import triton
import triton.language as tl


@triton.jit
def _bias_scale_kernel(x_ptr, out_ptr, bias_ptr, weight_ptr, n, HAS_BIAS: tl.constexpr,
                       HAS_WEIGHT: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    if HAS_BIAS:
        x = x + tl.load(bias_ptr)
    if HAS_WEIGHT:
        x = x * tl.load(weight_ptr)
    tl.store(out_ptr + offs, x, mask=mask)


class ScalarBiasScaleNew(nn.Module):
    def __init__(self, scale=True, scale_init=1.0, bias=True, bias_init=0.0) -> None:
        super().__init__()
        if scale:
            self.weight = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('weight', None)
        if bias:
            self.bias = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('bias', None)
        self.weight_init = scale_init
        self.bias_init = bias_init
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight is not None:
            init.constant_(self.weight, self.weight_init)
        if self.bias is not None:
            init.constant_(self.bias, self.bias_init)

    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = triton.next_power_of_2(n)
        _bias_scale_kernel[(1,)](
            x, out,
            self.bias if self.bias is not None else x,
            self.weight if self.weight is not None else x,
            n,
            self.bias is not None,
            self.weight is not None,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
        return out
