import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _noise_kernel(x_ptr, w_ptr, noise_ptr, out_ptr, n_elements, C, HW,
                  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    CHW = C * HW
    n = offs // CHW
    rem = offs % CHW
    c = rem // HW
    hw = rem % HW
    noise_idx = n * HW + hw
    x = tl.load(x_ptr + offs, mask=mask)
    w = tl.load(w_ptr + c, mask=mask)
    noise = tl.load(noise_ptr + noise_idx, mask=mask)
    tl.store(out_ptr + offs, x + w * noise, mask=mask)


class NoiseLayerNew(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
        self.noise = None

    def forward(self, x, noise=None):
        if noise is None and self.noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=
                x.device, dtype=x.dtype)
        elif noise is None:
            noise = self.noise
        N, C, H, W = x.shape
        x = x.contiguous()
        noise = noise.contiguous()
        out = torch.empty_like(x)
        n_elements = x.numel()
        HW = H * W
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _noise_kernel[grid](x, self.weight, noise, out, n_elements, C, HW,
                            BLOCK_SIZE=BLOCK_SIZE, num_warps=2)
        return out
