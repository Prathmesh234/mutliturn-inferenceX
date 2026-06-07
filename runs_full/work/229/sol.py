import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv3x3_kernel(x_ptr, w_ptr, b_ptr, res_ptr, out_ptr,
                    N, C: tl.constexpr, H, W,
                    APPLY_RELU: tl.constexpr, ADD_RES: tl.constexpr,
                    BLOCK: tl.constexpr):
    pid_nc = tl.program_id(0)
    pid_s = tl.program_id(1)
    n = pid_nc // C
    co = pid_nc % C
    HW = H * W
    offs = pid_s * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HW
    h = offs // W
    w = offs % W

    acc = tl.load(b_ptr + co).to(tl.float32) + tl.zeros((BLOCK,), tl.float32)

    base_in = n * C * HW
    for ci in range(C):
        in_c = base_in + ci * HW
        wbase = co * C * 9 + ci * 9
        for kh in range(3):
            ih = h + kh - 1
            for kw in range(3):
                iw = w + kw - 1
                valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                val = tl.load(x_ptr + in_c + ih * W + iw, mask=valid, other=0.0)
                wv = tl.load(w_ptr + wbase + kh * 3 + kw)
                acc += val * wv

    if APPLY_RELU:
        acc = tl.where(acc >= 0, acc, acc * 0.2)
    if ADD_RES:
        r = tl.load(res_ptr + base_in + co * HW + offs, mask=mask, other=0.0)
        acc += r

    tl.store(out_ptr + base_in + co * HW + offs, acc, mask=mask)


def _conv3x3(x, weight, bias, res, apply_relu, add_res):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    HW = H * W
    BLOCK = triton.next_power_of_2(HW)
    grid = (N * C, triton.cdiv(HW, BLOCK))
    _conv3x3_kernel[grid](
        x, weight, bias, res if res is not None else x, out,
        N, C, H, W, apply_relu, add_res, BLOCK=BLOCK, num_warps=4)
    return out


class _Residual_Block_SRNew(nn.Module):
    def __init__(self, num_ft):
        super(_Residual_Block_SRNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_ft, out_channels=num_ft,
            kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_ft, out_channels=num_ft,
            kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = x.contiguous()
        out = _conv3x3(x, self.conv1.weight, self.conv1.bias, None, True, False)
        out = _conv3x3(out, self.conv2.weight, self.conv2.bias, x, False, True)
        return out
