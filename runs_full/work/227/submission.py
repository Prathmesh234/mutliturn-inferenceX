import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                  N, C, HW, NC,
                  BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
                  BLOCK_HW: tl.constexpr, BLOCK_NC: tl.constexpr):
    offs_n = tl.arange(0, BLOCK_N)
    offs_c = tl.arange(0, BLOCK_C)
    offs_hw = tl.arange(0, BLOCK_HW)
    # x[N, C, HW]
    x = tl.load(x_ptr + offs_n[:, None, None] * (C * HW)
                + offs_c[None, :, None] * HW + offs_hw[None, None, :],
                mask=(offs_n[:, None, None] < N) & (offs_c[None, :, None] < C)
                & (offs_hw[None, None, :] < HW), other=0.0)
    pooled = tl.sum(x, axis=2) / HW  # [BLOCK_N, BLOCK_C]
    offs_o = tl.arange(0, BLOCK_NC)
    w = tl.load(w_ptr + offs_o[:, None] * C + offs_c[None, :],
                mask=(offs_o[:, None] < NC) & (offs_c[None, :] < C), other=0.0)
    # out[N, NC] = sum_c pooled[N,c]*w[NC,c]
    out = tl.sum(pooled[:, None, :] * w[None, :, :], axis=2)  # [BLOCK_N, BLOCK_NC]
    b = tl.load(b_ptr + offs_o, mask=offs_o < NC, other=0.0)
    out = out + b[None, :]
    tl.store(out_ptr + offs_n[:, None] * NC + offs_o[None, :], out,
             mask=(offs_n[:, None] < N) & (offs_o[None, :] < NC))


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
        _fused_kernel[(1,)](x, self.fc.weight, self.fc.bias, out,
                            N, C, HW, NC,
                            BLOCK_N=triton.next_power_of_2(N),
                            BLOCK_C=triton.next_power_of_2(C),
                            BLOCK_HW=triton.next_power_of_2(HW),
                            BLOCK_NC=triton.next_power_of_2(NC),
                            num_warps=1)
        return out
