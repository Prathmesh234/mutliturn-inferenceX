import torch
import torch.nn as nn
import triton
import triton.language as tl


def asymmetric_linear_quantization_scale_factor(num_bits, saturation_min, saturation_max):
    n = 2 ** num_bits - 1
    return n / (saturation_max - saturation_min)


@triton.jit
def _clq_kernel(x_ptr, out_ptr, n_elements, scale, clip_val, DEQ: tl.constexpr,
                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    x = tl.minimum(tl.maximum(x, 0.0), clip_val)
    q = tl.extra.cuda.libdevice.round(scale * x)
    if DEQ:
        q = q / scale
    tl.store(out_ptr + offs, q, mask=mask)


class ClippedLinearQuantizationNew(nn.Module):
    def __init__(self, num_bits, clip_val, dequantize=True, inplace=False):
        super().__init__()
        self.num_bits = num_bits
        self.clip_val = clip_val
        self.scale_factor = asymmetric_linear_quantization_scale_factor(num_bits, 0, clip_val)
        self.dequantize = dequantize
        self.inplace = inplace

    def forward(self, input):
        out = input if self.inplace else torch.empty_like(input)
        x = input.contiguous()
        n = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _clq_kernel[grid](x, out, n, float(self.scale_factor), float(self.clip_val),
                          self.dequantize, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
        return out
