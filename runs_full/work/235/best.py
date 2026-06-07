import torch
from torch import nn
from torch.nn.parameter import Parameter
import triton
import triton.language as tl


@triton.jit
def _tlu_kernel(x_ptr, tau_ptr, out_ptr, C, HW, BLOCK_HW: tl.constexpr):
    pid = tl.program_id(axis=0)  # over N*C
    c = pid % C
    tau = tl.load(tau_ptr + c)
    offs = pid * HW + tl.arange(0, BLOCK_HW)
    mask = tl.arange(0, BLOCK_HW) < HW
    x = tl.load(x_ptr + offs, mask=mask)
    out = tl.maximum(x, tau)
    tl.store(out_ptr + offs, out, mask=mask)


class TLUNew(nn.Module):
    def __init__(self, num_features):
        super(TLUNew, self).__init__()
        self.num_features = num_features
        self.tau = Parameter(torch.Tensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)

    def extra_repr(self):
        return 'num_features={num_features}'.format(**self.__dict__)

    def forward(self, x):
        x = x.contiguous()
        out = torch.empty_like(x)
        N, C, H, W = x.shape
        HW = H * W
        BLOCK_HW = triton.next_power_of_2(HW)
        grid = (N * C,)
        _tlu_kernel[grid](x, self.tau, out, C, HW, BLOCK_HW=BLOCK_HW, num_warps=1)
        return out
