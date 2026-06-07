import math
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter


class adaConv2d(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size:
        'int', stride: 'int'=1, padding: 'int'=0, dilation: 'int'=1, bias:
        'bool'=True):
        super(adaConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.weight = Parameter(torch.Tensor(out_channels, in_channels,
            kernel_size, kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def manual_conv(self, inp, kernel_size, stride, padding, dilation, scales):
        """
        :param inp: input feature map
        :param scales: scales for patches
        :return: new feature map
        """
        unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation,
            padding=padding, stride=stride)
        Hin = inp.shape[2]
        Win = inp.shape[3]
        w = self.weight
        Hout = math.floor((Hin + 2 * padding - dilation * (kernel_size - 1) -
            1) / stride + 1)
        Wout = math.floor((Win + 2 * padding - dilation * (kernel_size - 1) -
            1) / stride + 1)
        inp_unf = unfold(inp)
        n_boxes = inp_unf.shape[-1]
        inp_unf = inp_unf.view(inp.shape[0], inp.shape[1], kernel_size,
            kernel_size, n_boxes)
        inp_unf = inp_unf.permute(0, 4, 1, 2, 3)
        scales_unf = unfold(scales)
        scales_unf = scales_unf.view(scales.shape[0], scales.shape[1],
            kernel_size, kernel_size, scales_unf.shape[-1]).permute(0, 4, 1,
            2, 3)
        center_y, center_x = kernel_size // 2, kernel_size // 2
        scales_0 = torch.mean(scales_unf[:, :, :, center_y:center_y + 1,
            center_x:center_x + 1], axis=2, keepdim=True)
        scales_unf -= scales_0
        scales_unf = torch.exp(-0.5 * scales_unf * scales_unf)
        scales_unf = torch.mean(scales_unf, axis=2, keepdim=True)
        inp_unf *= scales_unf
        inp_unf = inp_unf.permute(0, 2, 3, 4, 1).view(inp.shape[0], inp.
            shape[1] * kernel_size * kernel_size, n_boxes)
        out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()
            ).transpose(1, 2)
        out_unf += self.bias.view(1, self.bias.shape[0], 1)
        output = out_unf.view(inp.shape[0], self.weight.shape[0], Hout, Wout)
        return output

    def reset_parameters(self) ->None:
        """
        init weight and bias
        :return:
        """
        nn.init.xavier_uniform(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0)

    def forward(self, input, scales=1):
        return self.manual_conv(input, self.kernel_size, self.stride, self.
            padding, self.dilation, scales=scales)


class adaModule(nn.Module):
    """
    paper module
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size:
        'int', stride: 'int'=1, padding: 'int'=0, dilation: 'int'=1):
        super(adaModule, self).__init__()
        self.conv = adaConv2d(in_channels, out_channels, kernel_size=
            kernel_size, dilation=dilation, padding=padding, stride=stride)

    def forward(self, input: 'Tensor', scales: 'Tensor') ->Tensor:
        return self.conv(input, scales=scales)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
