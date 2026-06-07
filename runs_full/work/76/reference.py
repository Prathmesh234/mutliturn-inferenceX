import torch
import torch.nn as nn


class SurfaceLoss(nn.Module):

    def __init__(self, epsilon=1e-05, softmax=True):
        super(SurfaceLoss, self).__init__()
        self.weight_map = []

    def forward(self, x, distmap):
        x = torch.softmax(x, dim=1)
        self.weight_map = distmap
        score = x.flatten(start_dim=2) * distmap.flatten(start_dim=2)
        score = torch.mean(score, dim=2)
        score = torch.mean(score, dim=1)
        return score


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
