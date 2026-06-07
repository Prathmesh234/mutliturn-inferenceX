import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import torch.onnx.operators
import torch.optim
import torch.optim.lr_scheduler


class Fp32GroupNorm(nn.GroupNorm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(input.float(), self.num_groups, self.weight.
            float() if self.weight is not None else None, self.bias.float() if
            self.bias is not None else None, self.eps)
        return output.type_as(input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_groups': 1, 'num_channels': 4}]
