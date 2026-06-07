import torch
import torch.nn.functional as F
import torch.nn as nn


def getIncomingShape(incoming):
    size = incoming.size()
    return [size[0], size[1], size[2], size[3]]


def interleave(tensors, axis):
    old_shape = getIncomingShape(tensors[0])[1:]
    new_shape = [-1] + old_shape
    new_shape[axis] *= len(tensors)
    stacked = torch.stack(tensors, axis + 1)
    reshaped = stacked.view(new_shape)
    return reshaped


class UnpoolingAsConvolution(nn.Module):

    def __init__(self, in_kernels, out_kernels):
        super(UnpoolingAsConvolution, self).__init__()
        self.conv_A = nn.Conv2d(in_kernels, out_kernels, kernel_size=(3, 3),
            stride=1, padding=1)
        self.conv_B = nn.Conv2d(in_kernels, out_kernels, kernel_size=(2, 3),
            stride=1, padding=0)
        self.conv_C = nn.Conv2d(in_kernels, out_kernels, kernel_size=(3, 2),
            stride=1, padding=0)
        self.conv_D = nn.Conv2d(in_kernels, out_kernels, kernel_size=(2, 2),
            stride=1, padding=0)

    def forward(self, x):
        out_a = self.conv_A(x)
        padded_b = F.pad(x, (1, 1, 0, 1))
        out_b = self.conv_B(padded_b)
        padded_c = F.pad(x, (0, 1, 1, 1))
        out_c = self.conv_C(padded_c)
        padded_d = F.pad(x, (0, 1, 0, 1))
        out_d = self.conv_D(padded_d)
        out_left = interleave([out_a, out_b], axis=2)
        out_right = interleave([out_c, out_d], axis=2)
        out = interleave([out_left, out_right], axis=3)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_kernels': 4, 'out_kernels': 4}]
