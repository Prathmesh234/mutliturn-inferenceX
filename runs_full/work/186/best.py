import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def conv3x3_relu_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, Cin, Cout, H, W,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    HW = H * W
    M = N * HW
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_idx = offs_m // HW
    rem = offs_m % HW
    oh = rem // W
    ow = rem % W
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kh in range(3):
        for kw in range(3):
            ih = oh + kh - 1
            iw = ow + kw - 1
            hw_mask = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            base = n_idx * (Cin * HW) + ih * W + iw
            for ci0 in range(0, Cin, BLOCK_K):
                offs_k = ci0 + tl.arange(0, BLOCK_K)
                k_mask = offs_k < Cin
                a_ptrs = x_ptr + base[:, None] + offs_k[None, :] * HW
                a_mask = m_mask[:, None] & hw_mask[:, None] & k_mask[None, :]
                a = tl.load(a_ptrs, mask=a_mask, other=0.0)
                w_off = offs_k[:, None] * 9 + (kh * 3 + kw)
                b_ptrs = w_ptr + offs_n[None, :] * (Cin * 9) + w_off
                b_mask = k_mask[:, None] & (offs_n[None, :] < Cout)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)
                acc += tl.dot(a, b)
    bias = tl.load(b_ptr + offs_n, mask=offs_n < Cout, other=0.0)
    acc += bias[None, :]
    acc = tl.maximum(acc, 0.0)
    y_ptrs = (y_ptr + n_idx[:, None] * (Cout * HW)
              + offs_n[None, :] * HW + (oh * W + ow)[:, None])
    y_mask = m_mask[:, None] & (offs_n[None, :] < Cout)
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def maxpool2x2_kernel(x_ptr, y_ptr, N, C, H, W, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    Ho = H // 2
    Wo = W // 2
    M = N * C * Ho * Wo
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    wo = offs % Wo
    ho = (offs // Wo) % Ho
    c = (offs // (Wo * Ho)) % C
    n = offs // (Wo * Ho * C)
    base = ((n * C + c) * H + ho * 2) * W + wo * 2
    ninf = float("-inf")
    x00 = tl.load(x_ptr + base, mask=mask, other=ninf)
    x01 = tl.load(x_ptr + base + 1, mask=mask, other=ninf)
    x10 = tl.load(x_ptr + base + W, mask=mask, other=ninf)
    x11 = tl.load(x_ptr + base + W + 1, mask=mask, other=ninf)
    out = tl.maximum(tl.maximum(x00, x01), tl.maximum(x10, x11))
    tl.store(y_ptr + offs, out, mask=mask)


def _conv_relu(x, weight, bias):
    N, Cin, H, W = x.shape
    Cout = weight.shape[0]
    x = x.contiguous()
    y = torch.empty((N, Cout, H, W), device=x.device, dtype=x.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(N * H * W, BLOCK_M), triton.cdiv(Cout, BLOCK_N))
    conv3x3_relu_kernel[grid](
        x, weight, bias, y, N, Cin, Cout, H, W,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=2,
    )
    return y


def _maxpool(x):
    N, C, H, W = x.shape
    y = torch.empty((N, C, H // 2, W // 2), device=x.device, dtype=x.dtype)
    M = y.numel()
    BLOCK = 256
    grid = (triton.cdiv(M, BLOCK),)
    maxpool2x2_kernel[grid](x, y, N, C, H, W, BLOCK=BLOCK, num_warps=4)
    return y


class Vgg16New(nn.Module):
    def __init__(self):
        super(Vgg16New, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def _cr(self, conv, x):
        return _conv_relu(x, conv.weight, conv.bias)

    def forward(self, X):
        h = self._cr(self.conv1_1, X)
        h = self._cr(self.conv1_2, h)
        h = _maxpool(h)
        h = self._cr(self.conv2_1, h)
        h = self._cr(self.conv2_2, h)
        h = _maxpool(h)
        h = self._cr(self.conv3_1, h)
        h = self._cr(self.conv3_2, h)
        h = self._cr(self.conv3_3, h)
        h = _maxpool(h)
        h = self._cr(self.conv4_1, h)
        h = self._cr(self.conv4_2, h)
        h = self._cr(self.conv4_3, h)
        h = self._cr(self.conv5_1, h)
        h = self._cr(self.conv5_2, h)
        h = self._cr(self.conv5_3, h)
        return h
