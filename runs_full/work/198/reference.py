import torch
import torch.nn as nn


class BCELoss2c(nn.Module):

    def __init__(self):
        super(BCELoss2c, self).__init__()
        self.bce0 = nn.BCEWithLogitsLoss()
        self.bce1 = nn.BCEWithLogitsLoss()
        None

    def forward(self, y_pred, y_true, weights=None):
        loss_0 = self.bce0(y_pred[:, 0], y_true[:, 0])
        loss_1 = self.bce1(y_pred[:, 1], y_true[:, 1])
        loss = loss_0 + loss_1
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
