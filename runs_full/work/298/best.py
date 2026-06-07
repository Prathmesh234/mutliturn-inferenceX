import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gate_kernel(x_ptr, g_ptr, out_ptr, C, HW, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    chan = (offs // HW) % C
    x = tl.load(x_ptr + offs)
    g = tl.load(g_ptr + chan)
    tl.store(out_ptr + offs, x * g)


class GateNew(nn.Module):
    def __init__(self, out_planes):
        super(GateNew, self).__init__()
        self.gate = nn.Parameter(torch.ones(1, out_planes, 1, 1), requires_grad=False)

    def forward(self, x):
        out = torch.empty_like(x)
        N, C, H, W = x.shape
        HW = H * W
        n = x.numel()
        g = self.gate.reshape(-1)
        BLOCK_SIZE = triton.next_power_of_2(n)
        _gate_kernel[(1,)](x, g, out, C, HW, BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
        return out
