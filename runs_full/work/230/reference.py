import torch
import torch.nn.functional
import torch.nn as nn


class _Residual_Block_DB(nn.Module):

    def __init__(self, num_ft):
        super(_Residual_Block_DB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_ft, out_channels=num_ft,
            kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_ft, out_channels=num_ft,
            kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output = torch.add(output, identity_data)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_ft': 4}]
