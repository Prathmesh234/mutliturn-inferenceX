import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3x3_relu_kernel(x_ptr, w_ptr, b_ptr, out_ptr, pool_ptr,
                        N, IC: tl.constexpr, OC: tl.constexpr,
                        H: tl.constexpr, W: tl.constexpr,
                        OH: tl.constexpr, OW: tl.constexpr,
                        BLOCK_HW: tl.constexpr, POOL: tl.constexpr):
    pid = tl.program_id(0)          # over N*OC
    n = pid // OC
    oc = pid % OC
    hw = tl.arange(0, BLOCK_HW)
    mask_hw = hw < (H * W)
    oh = hw // W
    ow = hw % W
    acc = tl.zeros((BLOCK_HW,), tl.float32)
    for ic in range(IC):
        for kh in range(3):
            for kw in range(3):
                ih = oh + kh - 1
                iw = ow + kw - 1
                valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & mask_hw
                in_off = ((n * IC + ic) * H + ih) * W + iw
                xval = tl.load(x_ptr + in_off, mask=valid, other=0.0)
                woff = ((oc * IC + ic) * 3 + kh) * 3 + kw
                acc += xval * tl.load(w_ptr + woff)
    acc += tl.load(b_ptr + oc)
    acc = tl.maximum(acc, 0.0)
    out_off = ((n * OC + oc) * H + oh) * W + ow
    tl.store(out_ptr + out_off, acc, mask=mask_hw)
    if POOL:
        img = tl.reshape(acc, (OH, 2, OW, 2))
        m = tl.max(tl.max(img, axis=3), axis=1)
        m = tl.reshape(m, (OH * OW,))
        ph = tl.arange(0, OH * OW)
        tl.store(pool_ptr + pid * OH * OW + ph, m)


def _conv_relu(x, weight, bias):
    N, IC, H, W = x.shape
    OC = weight.shape[0]
    out = torch.empty((N, OC, H, W), device=x.device, dtype=x.dtype)
    BLOCK_HW = triton.next_power_of_2(H * W)
    conv3x3_relu_kernel[(N * OC,)](x, weight, bias, out, out, N, IC, OC, H, W,
                                   1, 1, BLOCK_HW=BLOCK_HW, POOL=False, num_warps=1)
    return out


def _conv_relu_pool(x, weight, bias):
    N, IC, H, W = x.shape
    OC = weight.shape[0]
    OH, OW = H // 2, W // 2
    out = torch.empty((N, OC, H, W), device=x.device, dtype=x.dtype)
    pool = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_HW = triton.next_power_of_2(H * W)
    conv3x3_relu_kernel[(N * OC,)](x, weight, bias, out, pool, N, IC, OC, H, W,
                                   OH, OW, BLOCK_HW=BLOCK_HW, POOL=True, num_warps=1)
    return out, pool


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return _conv_relu(x.contiguous(), self.conv.weight, self.conv.bias)


class ConvRelu2(nn.Module):
    def __init__(self, _in, _out):
        super().__init__()
        self.cr1 = ConvRelu(_in, _out)
        self.cr2 = ConvRelu(_out, _out)

    def forward(self, x):
        return self.cr2(self.cr1(x))


class CoderNew(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv = ConvRelu2(in_size, out_size)
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, x):
        y1a = _conv_relu(x.contiguous(), self.conv.cr1.conv.weight, self.conv.cr1.conv.bias)
        y1, y2 = _conv_relu_pool(y1a, self.conv.cr2.conv.weight, self.conv.cr2.conv.bias)
        return y2, y1
