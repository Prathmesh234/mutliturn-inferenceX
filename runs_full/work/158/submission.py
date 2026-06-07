import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gab_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
                C, R, HW, BLOCK_C: tl.constexpr, BLOCK_R: tl.constexpr,
                BLOCK_HW: tl.constexpr):
    n = tl.program_id(0)
    c_idx = tl.arange(0, BLOCK_C)
    c_mask = c_idx < C
    r_idx = tl.arange(0, BLOCK_R)
    r_mask = r_idx < R
    hw_idx = tl.arange(0, BLOCK_HW)
    base = n * C * HW
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    n_chunks = tl.cdiv(HW, BLOCK_HW)
    for i in range(0, n_chunks):
        hw = i * BLOCK_HW + hw_idx
        hw_mask = hw < HW
        m = c_mask[:, None] & hw_mask[None, :]
        v = tl.load(x_ptr + base + c_idx[:, None] * HW + hw[None, :], mask=m, other=0.0)
        acc += tl.sum(v, axis=1)
    avg = acc / HW
    w1 = tl.load(w1_ptr + r_idx[:, None] * C + c_idx[None, :],
                 mask=r_mask[:, None] & c_mask[None, :], other=0.0)
    h = tl.sum(w1 * avg[None, :], axis=1) + tl.load(b1_ptr + r_idx, mask=r_mask, other=0.0)
    h = tl.maximum(h, 0.0)
    w2 = tl.load(w2_ptr + c_idx[:, None] * R + r_idx[None, :],
                 mask=c_mask[:, None] & r_mask[None, :], other=0.0)
    s = tl.sum(w2 * h[None, :], axis=1) + tl.load(b2_ptr + c_idx, mask=c_mask, other=0.0)
    s = tl.sigmoid(s)  # [BLOCK_C]
    for i in range(0, n_chunks):
        hw = i * BLOCK_HW + hw_idx
        hw_mask = hw < HW
        m = c_mask[:, None] & hw_mask[None, :]
        off = base + c_idx[:, None] * HW + hw[None, :]
        v = tl.load(x_ptr + off, mask=m, other=0.0)
        tl.store(out_ptr + off, v * s[:, None], mask=m)


class GABNew(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GABNew, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, H, W = x.shape
        R = self.conv1.out_channels
        HW = H * W
        x = x.contiguous()
        out = torch.empty_like(x)
        BLOCK_C = triton.next_power_of_2(C)
        BLOCK_R = triton.next_power_of_2(R)
        BLOCK_HW = triton.next_power_of_2(min(HW, 1024))
        w1 = self.conv1.weight.reshape(R, C).contiguous()
        w2 = self.conv2.weight.reshape(C, R).contiguous()
        _gab_kernel[(N,)](x, w1, self.conv1.bias, w2, self.conv2.bias, out,
                          C, R, HW, BLOCK_C=BLOCK_C, BLOCK_R=BLOCK_R,
                          BLOCK_HW=BLOCK_HW, num_warps=4)
        return out
