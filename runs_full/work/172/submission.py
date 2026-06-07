import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv_relu_kernel(x_ptr, w_ptr, b_ptr, p_ptr, out_ptr,
                      N, CIN, H, W, OC, OH, OW, K, PAD,
                      HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * OC * OH * OW
    mask = offs < total

    ow = offs % OW
    t = offs // OW
    oh = t % OH
    t = t // OH
    oc = t % OC
    n = t // OC

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for ic in range(0, CIN):
        for kh in range(0, K):
            ih = oh - PAD + kh
            for kw in range(0, K):
                iw = ow - PAD + kw
                valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                x_off = ((n * CIN + ic) * H + ih) * W + iw
                xval = tl.load(x_ptr + x_off, mask=mask & valid, other=0.0)
                w_off = ((oc * CIN + ic) * K + kh) * K + kw
                wval = tl.load(w_ptr + w_off, mask=mask, other=0.0)
                acc += xval * wval

    if HAS_BIAS:
        acc += tl.load(b_ptr + oc, mask=mask, other=0.0)

    p = tl.load(p_ptr)
    out = tl.where(acc >= 0, acc, p * acc)
    tl.store(out_ptr + offs, out, mask=mask)


class ConvReluNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(ConvReluNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, bias=bias)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = x.contiguous()
        N, CIN, H, W = x.shape
        OC = self.conv.out_channels
        K = self.conv.kernel_size[0]
        PAD = self.conv.padding[0]
        OH = H + 2 * PAD - K + 1
        OW = W + 2 * PAD - K + 1
        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)
        w = self.conv.weight
        has_bias = self.conv.bias is not None
        b = self.conv.bias if has_bias else w
        total = N * OC * OH * OW
        BLOCK = 128
        grid = (triton.cdiv(total, BLOCK),)
        _conv_relu_kernel[grid](x, w, b, self.relu.weight, out,
                                N, CIN, H, W, OC, OH, OW, K, PAD,
                                HAS_BIAS=has_bias, BLOCK=BLOCK, num_warps=2)
        return out
