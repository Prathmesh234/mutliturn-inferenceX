import torch
import torch.nn as nn
import torch.utils.checkpoint


class SelfAttention(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.fn = nn.MultiheadAttention(*args, **kwargs)

    def forward(self, x):
        x = torch.unsqueeze(x, -2)
        y, _ = self.fn(x, x, x, need_weights=False)
        return torch.squeeze(y, -2)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'embed_dim': 4, 'num_heads': 4}]
