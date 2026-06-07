import torch
from torch.optim.lr_scheduler import *
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.utils.model_zoo


def linear_dequantize(input, scale_factor, inplace=False):
    if inplace:
        input.div_(scale_factor)
        return input
    return input / scale_factor


def linear_quantize(input, scale_factor, inplace=False):
    if inplace:
        input.mul_(scale_factor).round_()
        return input
    return torch.round(scale_factor * input)


def asymmetric_linear_quantization_scale_factor(num_bits, saturation_min,
    saturation_max):
    n = 2 ** num_bits - 1
    return n / (saturation_max - saturation_min)


def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


class LinearQuantizeSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, scale_factor, dequantize, inplace):
        if inplace:
            ctx.mark_dirty(input)
        output = linear_quantize(input, scale_factor, inplace)
        if dequantize:
            output = linear_dequantize(output, scale_factor, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class ClippedLinearQuantization(nn.Module):

    def __init__(self, num_bits, clip_val, dequantize=True, inplace=False):
        super(ClippedLinearQuantization, self).__init__()
        self.num_bits = num_bits
        self.clip_val = clip_val
        self.scale_factor = asymmetric_linear_quantization_scale_factor(
            num_bits, 0, clip_val)
        self.dequantize = dequantize
        self.inplace = inplace

    def forward(self, input):
        input = clamp(input, 0, self.clip_val, self.inplace)
        input = LinearQuantizeSTE.apply(input, self.scale_factor, self.
            dequantize, self.inplace)
        return input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return '{0}(num_bits={1}, clip_val={2}{3})'.format(self.__class__.
            __name__, self.num_bits, self.clip_val, inplace_str)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_bits': 4, 'clip_val': 4}]
