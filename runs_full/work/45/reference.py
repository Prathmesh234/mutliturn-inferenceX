import torch
import torch.nn as nn
import torch.utils.data


class maxout(nn.Module):
    """
    maxout network
    """

    def __init__(self, in_feature, out_feature, pool_size):
        super(maxout, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.pool_size = pool_size
        self.linear = nn.Linear(in_feature, out_feature * pool_size)

    def forward(self, x):
        output = self.linear(x)
        output = output.view(-1, self.out_feature, self.pool_size)
        output = output.max(2)[0]
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_feature': 4, 'out_feature': 4, 'pool_size': 4}]
