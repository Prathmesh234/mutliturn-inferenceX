import torch
import torch.nn as nn


class MultiLevelPooling(nn.Module):

    def __init__(self, levels=[1, 2, 4]):
        super(MultiLevelPooling, self).__init__()
        self.Pools = nn.ModuleList([nn.MaxPool2d(i) for i in levels])

    def forward(self, x):
        assert len(x.size()) == 4, '输入形状不满足(n,c,w,w)'
        n = x.size(0)
        c = x.size(1)
        features = []
        for pool in self.Pools:
            features.append(pool(x))
        return features[0].view(n, c, -1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
