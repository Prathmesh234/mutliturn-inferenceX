import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _vflip_kernel(in_ptr, out_ptr, H, W, BLOCK_W: tl.constexpr):
    pid = tl.program_id(0)
    h = pid % H
    outer = pid // H
    row_base = outer * (H * W)
    src_h = H - 1 - h
    offs = tl.arange(0, BLOCK_W)
    mask = offs < W
    x = tl.load(in_ptr + row_base + src_h * W + offs, mask=mask)
    tl.store(out_ptr + row_base + h * W + offs, x, mask=mask)


class VflipNew(nn.Module):
    def __init__(self) -> None:
        super(VflipNew, self).__init__()

    def forward(self, input: 'torch.Tensor') -> torch.Tensor:
        x = input.contiguous()
        out = torch.empty_like(x)
        H = x.shape[-2]
        W = x.shape[-1]
        outer = x.numel() // (H * W)
        BLOCK_W = triton.next_power_of_2(W)
        grid = (outer * H,)
        _vflip_kernel[grid](x, out, H, W, BLOCK_W=BLOCK_W, num_warps=1)
        return out

    def __repr__(self):
        return self.__class__.__name__
