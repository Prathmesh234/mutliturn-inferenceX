import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rgba_to_rgb_kernel(in_ptr, out_ptr, CHW, IHW, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    b = offs // CHW
    rem = offs % CHW
    x = tl.load(in_ptr + b * IHW + rem, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)


class RgbaToRgbNew(nn.Module):
    def __init__(self) -> None:
        super(RgbaToRgbNew, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if len(image.shape) < 3 or image.shape[-3] != 4:
            raise ValueError(
                f'Input size must have a shape of (*, 4, H, W).Got {image.shape}')
        image = image.contiguous()
        H = image.shape[-2]
        W = image.shape[-1]
        HW = H * W
        CHW = 3 * HW
        IHW = 4 * HW
        out_shape = list(image.shape)
        out_shape[-3] = 3
        out = torch.empty(out_shape, device=image.device, dtype=image.dtype)
        n_elements = out.numel()
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _rgba_to_rgb_kernel[grid](image, out, CHW, IHW, n_elements,
                                  BLOCK_SIZE=BLOCK_SIZE, num_warps=2)
        return out
