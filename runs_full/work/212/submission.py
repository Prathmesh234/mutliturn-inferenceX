import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _avgpoolpad_kernel(x_ptr, out_ptr, N, C, H, W, OH, OW, stride,
                       total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < total

    fw = idx % OW
    t = idx // OW
    fh = t % OH
    t = t // OH
    c = t % C
    n = t // C

    oh = fh + 1
    ow = fw + 1

    base = n * C * H * W + c * H * W

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    cnt = tl.zeros([BLOCK], dtype=tl.float32)

    for kr in tl.static_range(3):
        for kc in tl.static_range(3):
            r = oh * stride - 1 + kr
            col = ow * stride - 1 + kc
            valid_r = (r >= 0) & (r <= H)
            valid_c = (col >= 0) & (col <= W)
            inrange = valid_r & valid_c
            loadable = inrange & (r >= 1) & (col >= 1)
            xoff = base + (r - 1) * W + (col - 1)
            val = tl.load(x_ptr + xoff, mask=mask & loadable, other=0.0)
            acc += val
            cnt += tl.where(inrange, 1.0, 0.0)

    res = acc / cnt
    tl.store(out_ptr + idx, res, mask=mask)


class AvgPoolPadNew(nn.Module):

    def __init__(self, stride=2, padding=1):
        super(AvgPoolPadNew, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.AvgPool2d(3, stride=stride, padding=padding,
            count_include_pad=False)
        self.stride = stride

    def forward(self, x):
        x = x.contiguous()
        N, C, H, W = x.shape
        Hp, Wp = H + 1, W + 1
        Ho = (Hp - 1) // self.stride + 1
        Wo = (Wp - 1) // self.stride + 1
        OH, OW = Ho - 1, Wo - 1
        out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
        total = N * C * OH * OW
        BLOCK = 256
        grid = (triton.cdiv(total, BLOCK),)
        _avgpoolpad_kernel[grid](x, out, N, C, H, W, OH, OW, self.stride,
                                 total, BLOCK=BLOCK, num_warps=1)
        return out
