import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import torch.onnx.operators
import torch.optim
import torch.optim.lr_scheduler


class RobertaMaskLeanerHead(nn.Module):
    """
    Head for mask leaner.
    input: (batch, src_lens, embed_dim)
    output: (batch, src_lens，1)
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.dense = nn.Linear(embed_dim, 1)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = x.view(x.size(0), -1)
        x = F.softmax(x, dim=-1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'embed_dim': 4}]
