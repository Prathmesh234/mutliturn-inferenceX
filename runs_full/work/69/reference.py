import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


class BiAttention(nn.Module):

    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)
        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 /
            input_size ** 0.5))

    def forward(self, input, memory, mask=None):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(
            1)
        input = self.dropout(input)
        memory = self.dropout(memory)
        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 
            1).contiguous())
        att = input_dot + memory_dot + cross_dot
        if mask is not None:
            att = att - 1e+30 * (1 - mask[:, None])
        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1,
            input_len)
        output_two = torch.bmm(weight_two, input)
        return torch.cat([input, output_one, input * output_one, output_two *
            output_one], dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'dropout': 0.5}]
