import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import triton
import triton.language as tl


@triton.jit
def _ada_kernel(
    inp_ptr, scl_ptr, w_ptr, b_ptr, out_ptr,
    N, C, H, W, O, Hout, Wout, n_boxes,
    K: tl.constexpr, STRIDE: tl.constexpr, PAD: tl.constexpr, DIL: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_O: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // n_boxes
    box = pid % n_boxes
    oh = box // Wout
    ow = box % Wout

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    offs_o = tl.arange(0, BLOCK_O)
    mask_o = offs_o < O

    inp_base = n * C * H * W
    scl_base = n * C * H * W
    chw = H * W

    cy = K // 2
    cx = K // 2
    ih_c = oh * STRIDE - PAD + cy * DIL
    iw_c = ow * STRIDE - PAD + cx * DIL
    cin = (ih_c >= 0) and (ih_c < H) and (iw_c >= 0) and (iw_c < W)
    c_ptr = scl_ptr + scl_base + offs_c * chw + ih_c * W + iw_c
    s0v = tl.load(c_ptr, mask=mask_c & cin, other=0.0)
    s0 = tl.sum(tl.where(mask_c, s0v, 0.0)) / C

    acc = tl.zeros((BLOCK_O,), dtype=tl.float32)

    for ki in tl.static_range(K):
        for kj in tl.static_range(K):
            ih = oh * STRIDE - PAD + ki * DIL
            iw = ow * STRIDE - PAD + kj * DIL
            inb = (ih >= 0) and (ih < H) and (iw >= 0) and (iw < W)
            off = offs_c * chw + ih * W + iw
            sp = tl.load(scl_ptr + scl_base + off, mask=mask_c & inb, other=0.0)
            d = sp - s0
            e = tl.exp(-0.5 * d * d)
            sf = tl.sum(tl.where(mask_c, e, 0.0)) / C

            xp = tl.load(inp_ptr + inp_base + off, mask=mask_c & inb, other=0.0)
            scaled = xp * sf  # [BLOCK_C]

            woff = offs_o[:, None] * (C * K * K) + offs_c[None, :] * (K * K) + ki * K + kj
            wmask = mask_o[:, None] & mask_c[None, :]
            w2d = tl.load(w_ptr + woff, mask=wmask, other=0.0)
            acc += tl.sum(w2d * scaled[None, :], axis=1)

    bias = tl.load(b_ptr + offs_o, mask=mask_o, other=0.0)
    acc += bias
    out_off = n * O * n_boxes + offs_o * n_boxes + box
    tl.store(out_ptr + out_off, acc, mask=mask_o)


class adaConv2dNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True):
        super(adaConv2dNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.weight = Parameter(torch.Tensor(out_channels, in_channels,
                                              kernel_size, kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0)

    def forward(self, input, scales=1):
        k = self.kernel_size
        N, C, H, W = input.shape
        O = self.out_channels
        Hout = (H + 2 * self.padding - self.dilation * (k - 1) - 1) // self.stride + 1
        Wout = (W + 2 * self.padding - self.dilation * (k - 1) - 1) // self.stride + 1
        n_boxes = Hout * Wout
        inp = input.contiguous()
        scl = scales.contiguous()
        w = self.weight.contiguous()
        b = self.bias.contiguous()
        out = torch.empty((N, O, Hout, Wout), device=input.device, dtype=input.dtype)
        BLOCK_C = triton.next_power_of_2(C)
        BLOCK_O = triton.next_power_of_2(O)
        grid = (N * n_boxes,)
        _ada_kernel[grid](
            inp, scl, w, b, out,
            N, C, H, W, O, Hout, Wout, n_boxes,
            K=k, STRIDE=self.stride, PAD=self.padding, DIL=self.dilation,
            BLOCK_C=BLOCK_C, BLOCK_O=BLOCK_O, num_warps=4,
        )
        return out


class adaModuleNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1):
        super(adaModuleNew, self).__init__()
        self.conv = adaConv2dNew(in_channels, out_channels, kernel_size=kernel_size,
                                 dilation=dilation, padding=padding, stride=stride)

    def forward(self, input, scales):
        return self.conv(input, scales=scales)
