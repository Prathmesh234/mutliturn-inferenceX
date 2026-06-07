import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _reduce_kernel(x_ptr, y_ptr, HW, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < HW
    x = tl.load(x_ptr + pid * HW + offs, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)
    mean = s / HW
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / HW
    std = tl.sqrt(var)
    tl.store(y_ptr + pid, std + mean)


@triton.jit
def _gemv_kernel(x_ptr, w_ptr, out_ptr, K, M, ACT: tl.constexpr,
                 BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    x = tl.load(x_ptr + pid_n * K + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + pid_m * K + offs, mask=mask, other=0.0)
    acc = tl.sum(x * w, axis=0)
    if ACT == 1:
        acc = tl.maximum(acc, 0.0)
    elif ACT == 2:
        acc = 1.0 / (1.0 + tl.exp(-acc))
    tl.store(out_ptr + pid_n * M + pid_m, acc)


class LCCALayerNew(nn.Module):

    def __init__(self, channel):
        super(LCCALayerNew, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c3 = nn.Conv2d(channel, channel // 4, kernel_size=3, padding=1,
                            bias=False)
        self.c32 = nn.Conv2d(channel // 4, channel, kernel_size=3, padding=1,
                             bias=False)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, H, W = x.shape
        HW = H * W
        xc = x.contiguous().view(N * C, HW)
        y = torch.empty(N * C, device=x.device, dtype=x.dtype)
        BLOCK = triton.next_power_of_2(HW)
        _reduce_kernel[(N * C,)](xc, y, HW, BLOCK=BLOCK, num_warps=4)
        y = y.view(N, C)

        C1 = C // 4
        W3c = self.c3.weight[:, :, 1, 1].contiguous()   # [C1, C]
        W32c = self.c32.weight[:, :, 1, 1].contiguous() # [C, C1]

        out1 = torch.empty(N, C1, device=x.device, dtype=x.dtype)
        _gemv_kernel[(N, C1)](y, W3c, out1, C, C1, 1,
                              BLOCK_K=triton.next_power_of_2(C), num_warps=4)

        out2 = torch.empty(N, C, device=x.device, dtype=x.dtype)
        _gemv_kernel[(N, C)](out1, W32c, out2, C1, C, 2,
                             BLOCK_K=triton.next_power_of_2(C1), num_warps=4)

        return out2.view(N, C, 1, 1)
