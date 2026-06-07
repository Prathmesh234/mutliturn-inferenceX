import torch
import torch.nn as nn


class CatConv(nn.Module):

    def __init__(self, in_kernels_1, in_kernels_2, kernels):
        super(CatConv, self).__init__()
        self.conv = nn.Conv2d(in_kernels_1 + in_kernels_2, kernels,
            kernel_size=1, bias=True)

    def forward(self, x1, x2):
        x1 = torch.cat([x2, x1], dim=1)
        x1 = self.conv(x1)
        return x1


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_kernels_1': 4, 'in_kernels_2': 4, 'kernels': 4}]
