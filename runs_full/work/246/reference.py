import torch
import torch.nn as nn


class MyUpsample2(nn.Module):

    def forward(self, x):
        return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(x
            .size(0), x.size(1), x.size(2) * 2, x.size(3) * 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
