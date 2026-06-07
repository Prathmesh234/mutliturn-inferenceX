import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _se_kernel(
    inp_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
    C, SQ, HW,
    C_BLK: tl.constexpr, SQ_BLK: tl.constexpr, HW_BLK: tl.constexpr,
):
    n = tl.program_id(0)
    offs_c = tl.arange(0, C_BLK)
    offs_sq = tl.arange(0, SQ_BLK)
    mask_c = offs_c < C
    mask_sq = offs_sq < SQ
    base = n * C * HW

    # average pool over HW per channel
    acc = tl.zeros((C_BLK,), dtype=tl.float32)
    for hw_start in range(0, HW, HW_BLK):
        offs_hw = hw_start + tl.arange(0, HW_BLK)
        mask_hw = offs_hw < HW
        ptr = inp_ptr + base + offs_c[:, None] * HW + offs_hw[None, :]
        vals = tl.load(ptr, mask=mask_c[:, None] & mask_hw[None, :], other=0.0)
        acc += tl.sum(vals, axis=1)
    avg = acc / HW

    # fc1 + relu
    w1 = tl.load(w1_ptr + offs_sq[:, None] * C + offs_c[None, :],
                 mask=mask_sq[:, None] & mask_c[None, :], other=0.0)
    s1 = tl.sum(w1 * avg[None, :], axis=1)
    s1 = s1 + tl.load(b1_ptr + offs_sq, mask=mask_sq, other=0.0)
    s1 = tl.maximum(s1, 0.0)

    # fc2
    w2 = tl.load(w2_ptr + offs_c[:, None] * SQ + offs_sq[None, :],
                 mask=mask_c[:, None] & mask_sq[None, :], other=0.0)
    s2 = tl.sum(w2 * s1[None, :], axis=1)
    s2 = s2 + tl.load(b2_ptr + offs_c, mask=mask_c, other=0.0)

    # hardsigmoid
    hs = tl.minimum(tl.maximum(s2 + 3.0, 0.0), 6.0) / 6.0  # [C_BLK]

    # apply scale
    for hw_start in range(0, HW, HW_BLK):
        offs_hw = hw_start + tl.arange(0, HW_BLK)
        mask_hw = offs_hw < HW
        m = mask_c[:, None] & mask_hw[None, :]
        ptr = base + offs_c[:, None] * HW + offs_hw[None, :]
        x = tl.load(inp_ptr + ptr, mask=m, other=0.0)
        tl.store(out_ptr + ptr, x * hs[:, None], mask=m)


def _next_pow2(x):
    return 1 << (x - 1).bit_length()


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcitationNew(nn.Module):
    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def forward(self, input):
        N, C, H, W = input.shape
        HW = H * W
        SQ = self.fc1.weight.shape[0]
        inp = input.contiguous()
        out = torch.empty_like(inp)

        w1 = self.fc1.weight.reshape(SQ, C)
        w2 = self.fc2.weight.reshape(C, SQ)

        C_BLK = _next_pow2(C)
        SQ_BLK = _next_pow2(SQ)
        HW_BLK = min(2048, _next_pow2(HW))

        _se_kernel[(N,)](
            inp, w1, self.fc1.bias, w2, self.fc2.bias, out,
            C, SQ, HW,
            C_BLK=C_BLK, SQ_BLK=SQ_BLK, HW_BLK=HW_BLK, num_warps=1,
        )
        return out
