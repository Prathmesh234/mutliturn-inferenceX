import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                  C, HW, NC,
                  BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr, BLOCK_NC: tl.constexpr):
    n = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    offs_hw = tl.arange(0, BLOCK_HW)
    x = tl.load(x_ptr + n * C * HW + offs_c[:, None] * HW + offs_hw[None, :],
                mask=(offs_c[:, None] < C) & (offs_hw[None, :] < HW), other=0.0)
    pooled = tl.sum(x, axis=1) / HW
    offs_o = tl.arange(0, BLOCK_NC)
    w = tl.load(w_ptr + offs_o[:, None] * C + offs_c[None, :],
                mask=(offs_o[:, None] < NC) & (offs_c[None, :] < C), other=0.0)
    out = tl.sum(w * pooled[None, :], axis=1)
    b = tl.load(b_ptr + offs_o, mask=offs_o < NC, other=0.0)
    out = out + b
    tl.store(out_ptr + n * NC + offs_o, out, mask=offs_o < NC)


class AnyHeadNew(nn.Module):
    def __init__(self, w_in, nc):
        super(AnyHeadNew, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        N, C, H, W = x.shape
        HW = H * W
        x = x.contiguous()
        NC = self.fc.out_features
        out = torch.empty((N, NC), device=x.device, dtype=x.dtype)
        BLOCK_C = triton.next_power_of_2(C)
        BLOCK_HW = triton.next_power_of_2(HW)
        BLOCK_NC = triton.next_power_of_2(NC)
        _fused_kernel[(N,)](x, self.fc.weight, self.fc.bias, out,
                            C, HW, NC,
                            BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW, BLOCK_NC=BLOCK_NC,
                            num_warps=1)
        return out
