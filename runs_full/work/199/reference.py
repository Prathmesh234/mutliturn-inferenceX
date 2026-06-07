import torch
import torch.nn as nn


class BCELoss(nn.Module):

    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true, weights=None):
        loss_0 = self.bce(y_pred[:, 0], y_true[:, 0])
        loss_1 = self.bce(y_pred[:, 1], y_true[:, 1])
        loss_2 = self.bce(y_pred[:, 2], y_true[:, 2])
        loss_3 = self.bce(y_pred[:, 3], y_true[:, 3])
        loss = loss_0 * 0.1 + loss_1 * 0.5 + loss_2 * 0.3 + loss_3 * 0.3
        return loss * 100


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
