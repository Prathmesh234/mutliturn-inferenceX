import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv3x3_kernel(x_ptr, w_ptr, b_ptr, res_ptr, out_ptr,
                    N, H, W,
                    C: tl.constexpr, BLOCK: tl.constexpr,
                    APPLY_RELU: tl.constexpr, ADD_RES: tl.constexpr):
    pid_noc = tl.program_id(0)
    pid_sp = tl.program_id(1)
    n = pid_noc // C
    oc = pid_noc % C

    offs = pid_sp * BLOCK + tl.arange(0, BLOCK)
    HW = H * W
    mask = offs < HW
    oh = offs // W
    ow = offs % W

    acc = tl.load(b_ptr + oc).to(tl.float32) + tl.zeros((BLOCK,), tl.float32)

    base_n = n * C * HW
    for ic in tl.static_range(C):
        wbase = oc * C * 9 + ic * 9
        in_base = base_n + ic * HW
        for kh in tl.static_range(3):
            ih = oh + kh - 1
            vh = (ih >= 0) & (ih < H)
            for kw in tl.static_range(3):
                iw = ow + kw - 1
                valid = vh & (iw >= 0) & (iw < W) & mask
                ptr = in_base + ih * W + iw
                val = tl.load(x_ptr + ptr, mask=valid, other=0.0).to(tl.float32)
                wv = tl.load(w_ptr + wbase + kh * 3 + kw).to(tl.float32)
                acc += val * wv

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    out_off = base_n + oc * HW + offs
    if ADD_RES:
        r = tl.load(res_ptr + out_off, mask=mask, other=0.0).to(tl.float32)
        acc += r

    tl.store(out_ptr + out_off, acc, mask=mask)


def _conv3x3(x, weight, bias, relu, res):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(H * W)
    grid = (N * C, triton.cdiv(H * W, BLOCK))
    _conv3x3_kernel[grid](
        x, weight, bias, res if res is not None else x, out,
        N, H, W, C=C, BLOCK=BLOCK,
        APPLY_RELU=relu, ADD_RES=(res is not None),
        num_warps=4,
    )
    return out


class _Residual_Block_DBNew(nn.Module):
    def __init__(self, num_ft):
        super(_Residual_Block_DBNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_ft, out_channels=num_ft,
            kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_ft, out_channels=num_ft,
            kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = x.contiguous()
        out = _conv3x3(x, self.conv1.weight, self.conv1.bias, True, None)
        out = _conv3x3(out, self.conv2.weight, self.conv2.bias, False, x)
        return out
