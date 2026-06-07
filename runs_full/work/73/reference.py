import torch
from torch import nn


class GroupScaling1D(nn.Module):
    """Scales inputs by the second moment for the entire layer."""

    def __init__(self, eps=1e-05, group_num=4):
        super(GroupScaling1D, self).__init__()
        self.eps = eps
        self.group_num = group_num

    def extra_repr(self):
        return f'eps={self.eps}, group={self.group_num}'

    def forward(self, input):
        T, B, C = input.shape[0], input.shape[1], input.shape[2]
        Cg = C // self.group_num
        gn_input = input.contiguous().reshape(T, B, self.group_num, Cg)
        moment2 = torch.repeat_interleave(torch.mean(gn_input * gn_input,
            dim=3, keepdim=True), repeats=Cg, dim=-1).contiguous().reshape(T,
            B, C)
        return input / torch.sqrt(moment2 + self.eps)


def get_inputs():
    return [torch.rand([4, 4, 4, 1])]


def get_init_inputs():
    return [[], {}]
