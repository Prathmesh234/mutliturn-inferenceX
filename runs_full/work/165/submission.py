import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _noise_gen_kernel(img_ptr, w_ptr, out_ptr, n, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    img = tl.load(img_ptr + offs, mask=mask)
    noise = tl.randn(seed, offs)
    w = tl.load(w_ptr)
    tl.store(out_ptr + offs, img + w * noise, mask=mask)


@triton.jit
def _noise_kernel(img_ptr, noise_ptr, w_ptr, out_ptr, n, CHW, HW,
                  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    img = tl.load(img_ptr + offs, mask=mask)
    b = offs // CHW
    hw = offs % HW
    nidx = b * HW + hw
    noise = tl.load(noise_ptr + nidx, mask=mask)
    w = tl.load(w_ptr)
    tl.store(out_ptr + offs, img + w * noise, mask=mask)


class NoiseInjectionNew(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        out = torch.empty_like(image)
        n = image.numel()
        if noise is None:
            BLOCK_SIZE = triton.next_power_of_2(n)
            grid = (1,)
            _noise_gen_kernel[grid](image, self.weight, out, n, 0,
                                    BLOCK_SIZE=BLOCK_SIZE, num_warps=1)
        else:
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n, BLOCK_SIZE),)
            batch, C, height, width = image.shape
            HW = height * width
            CHW = C * HW
            _noise_kernel[grid](image, noise, self.weight, out, n, CHW, HW,
                                BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out
