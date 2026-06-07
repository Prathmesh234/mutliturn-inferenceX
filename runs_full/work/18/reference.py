import torch
import torch.nn as nn


class TripletLoss(nn.Module):

    def __init__(self):
        super(TripletLoss, self).__init__()
        self.margin = 0.5

    def distance(self, x, y):
        diff = torch.abs(x - y)
        diff = torch.pow(diff, 2).sum(-1)
        return diff

    def forward(self, anchor, pos, neg):
        pos_distance = self.distance(anchor, pos)
        neg_distance = self.distance(anchor, neg)
        loss = torch.clamp(self.margin + pos_distance - neg_distance, min=0.0
            ).mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
