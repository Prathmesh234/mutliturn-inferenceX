import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rgba_to_bgr_kernel(in_ptr, out_ptr, HW, BLOCK_SIZE: tl.constexpr):
    plane = tl.program_id(axis=0)   # n*3 + c_out
    blk = tl.program_id(axis=1)
    n = plane // 3
    c_out = plane % 3
    c_in = 2 - c_out
    in_base = (n * 4 + c_in) * HW
    out_base = plane * HW
    offs = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < HW
    x = tl.load(in_ptr + in_base + offs, mask=mask)
    tl.store(out_ptr + out_base + offs, x, mask=mask)


class RgbaToBgrNew(nn.Module):
    def __init__(self) -> None:
        super(RgbaToBgrNew, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if len(image.shape) < 3 or image.shape[-3] != 4:
            raise ValueError(f'Input size must have a shape of (*, 4, H, W).Got {image.shape}')
        image = image.contiguous()
        *batch, C, H, W = image.shape
        N = 1
        for b in batch:
            N *= b
        HW = H * W
        out = torch.empty((*batch, 3, H, W), dtype=image.dtype, device=image.device)
        BLOCK_SIZE = min(triton.next_power_of_2(HW), 1024)
        grid = (N * 3, triton.cdiv(HW, BLOCK_SIZE))
        _rgba_to_bgr_kernel[grid](image, out, HW, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out
