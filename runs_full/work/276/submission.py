import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.nn.modules.utils import _pair


@triton.jit
def _patch_kernel(in_ptr, out_ptr, total,
                  C, H, W,
                  nph, npw, wh, ww,
                  sh, sw, ph, pw,
                  BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    j = offs % ww
    t = offs // ww
    i = t % wh
    t = t // wh
    c = t % C
    t = t // C
    NN = nph * npw
    n = t % NN
    b = t // NN

    p_h = n // npw
    p_w = n % npw

    row = p_h * sh + i - ph
    col = p_w * sw + j - pw

    valid = mask & (row >= 0) & (row < H) & (col >= 0) & (col < W)
    in_idx = ((b * C + c) * H + row) * W + col
    val = tl.load(in_ptr + in_idx, mask=valid, other=0.0)
    tl.store(out_ptr + offs, val, mask=mask)


class ExtractTensorPatchesNew(nn.Module):
    def __init__(self, window_size, stride=1, padding=0):
        super().__init__()
        self.window_size = _pair(window_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

    def forward(self, input):
        input = input.contiguous()
        B, C, H, W = input.shape
        wh, ww = self.window_size
        sh, sw = self.stride
        ph, pw = self.padding
        nph = (H + 2 * ph - wh) // sh + 1
        npw = (W + 2 * pw - ww) // sw + 1
        N = nph * npw
        out = torch.empty((B, N, C, wh, ww), device=input.device, dtype=input.dtype)
        total = out.numel()
        if total == 0:
            return out
        BLOCK = min(triton.next_power_of_2(total), 1024)
        grid = (triton.cdiv(total, BLOCK),)
        _patch_kernel[grid](input, out, total,
                            C, H, W,
                            nph, npw, wh, ww,
                            sh, sw, ph, pw,
                            BLOCK=BLOCK, num_warps=4)
        return out
