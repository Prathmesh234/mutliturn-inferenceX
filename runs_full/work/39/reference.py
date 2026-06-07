import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data


class SparseDownSampleClose(nn.Module):

    def __init__(self, stride):
        super(SparseDownSampleClose, self).__init__()
        self.pooling = nn.MaxPool2d(stride, stride)
        self.large_number = 600

    def forward(self, d, mask):
        encode_d = -(1 - mask) * self.large_number - d
        d = -self.pooling(encode_d)
        mask_result = self.pooling(mask)
        d_result = d - (1 - mask_result) * self.large_number
        return d_result, mask_result


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'stride': 1}]
