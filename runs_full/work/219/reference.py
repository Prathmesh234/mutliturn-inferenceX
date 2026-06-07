import torch
import torch.nn as nn


class WeightedBCELoss(nn.Module):

    def __init__(self):
        super(WeightedBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, y_pred, y_true, weights):
        loss_0 = self.bce(y_pred[:, 0], y_true[:, 0])
        loss_1 = self.bce(y_pred[:, 1], y_true[:, 1])
        loss_2 = self.bce(y_pred[:, 2], y_true[:, 2])
        loss_3 = self.bce(y_pred[:, 3], y_true[:, 3])
        loss = (loss_0 + loss_1 + loss_2 + loss_3) * weights
        return loss.mean() * 10


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
