import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import triton
import triton.language as tl


@triton.jit
def _ssb_kernel(x_ptr, w_ptr, b_ptr, out_ptr, n_elements,
                HAS_W: tl.constexpr, HAS_B: tl.constexpr,
                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    if HAS_W:
        x = x * tl.load(w_ptr)
    if HAS_B:
        x = x + tl.load(b_ptr)
    tl.store(out_ptr + offs, x, mask=mask)


class ScalarScaleBiasNew(nn.Module):

    def __init__(self, scale=True, scale_init=1.0, bias=True, bias_init=0.0) -> None:
        super(ScalarScaleBiasNew, self).__init__()
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
        self._has_w = self.weight is not None
        self._has_b = self.bias is not None

    def reset_parameters(self) -> None:
        if self.weight is not None:
            init.constant_(self.weight, self.weight_init)
        if self.bias is not None:
            init.constant_(self.bias, self.bias_init)

    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _ssb_kernel[grid](x, self.weight, self.bias, out, n,
                          self._has_w, self._has_b,
                          BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out
