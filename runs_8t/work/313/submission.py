import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _catconv_kernel(x1_ptr, x2_ptr, w_ptr, b_ptr, out_ptr,
                    N, HW, C1: tl.constexpr, C2: tl.constexpr,
                    CI: tl.constexpr, CO: tl.constexpr,
                    BLOCK_CO: tl.constexpr, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    M = N * HW
    m_mask = offs_m < M
    n = offs_m // HW
    hw = offs_m % HW

    co_offs = tl.arange(0, BLOCK_CO)
    co_mask = co_offs < CO

    acc = tl.zeros((BLOCK_M, BLOCK_CO), tl.float32)
    for ci in tl.static_range(CI):
        if ci < C2:
            ptr = x2_ptr + n * (C2 * HW) + ci * HW + hw
        else:
            ci1 = ci - C2
            ptr = x1_ptr + n * (C1 * HW) + ci1 * HW + hw
        val = tl.load(ptr, mask=m_mask, other=0.0)
        w_col = tl.load(w_ptr + co_offs * CI + ci, mask=co_mask, other=0.0)
        acc += val[:, None] * w_col[None, :]

    bias = tl.load(b_ptr + co_offs, mask=co_mask, other=0.0)
    acc += bias[None, :]

    out_idx = n[:, None] * (CO * HW) + hw[:, None] + co_offs[None, :] * HW
    mask = m_mask[:, None] & co_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=mask)


class CatConvNew(nn.Module):
    def __init__(self, in_kernels_1, in_kernels_2, kernels):
        super(CatConvNew, self).__init__()
        self.conv = nn.Conv2d(in_kernels_1 + in_kernels_2, kernels,
                              kernel_size=1, bias=True)
        self.in_kernels_1 = in_kernels_1
        self.in_kernels_2 = in_kernels_2

    def forward(self, x1, x2):
        x1 = x1.contiguous()
        x2 = x2.contiguous()
        N, C1, H, W = x1.shape
        C2 = x2.shape[1]
        HW = H * W
        CO = self.conv.out_channels
        out = torch.empty((N, CO, H, W), device=x1.device, dtype=x1.dtype)

        w = self.conv.weight.reshape(CO, C1 + C2)
        b = self.conv.bias

        M = N * HW
        BLOCK_M = 32
        BLOCK_CO = triton.next_power_of_2(CO)
        grid = (triton.cdiv(M, BLOCK_M),)
        _catconv_kernel[grid](x1, x2, w, b, out,
                              N, HW, self.in_kernels_1, self.in_kernels_2,
                              C1 + C2, CO, BLOCK_CO, BLOCK_M, num_warps=4)
        return out
