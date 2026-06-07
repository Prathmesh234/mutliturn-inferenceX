import torch
import torch.nn as nn
import torch.nn.functional as F


class MCCLoss(nn.Module):

    def __init__(self, eps=1e-06):
        super(MCCLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true, w=None):
        y_pred = F.softmax(y_pred, dim=1)
        y_true_mean = torch.mean(y_true)
        y_pred_mean = torch.mean(y_pred)
        y_true_var = torch.var(y_true)
        y_pred_var = torch.var(y_pred)
        y_true_std = torch.std(y_true)
        y_pred_std = torch.std(y_pred)
        vx = y_true - torch.mean(y_true)
        vy = y_pred - torch.mean(y_pred)
        pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + self.
            eps) * torch.sqrt(torch.sum(vy ** 2) + self.eps))
        ccc = 2 * pcc * y_true_std * y_pred_std / (y_true_var + y_pred_var +
            (y_pred_mean - y_true_mean) ** 2)
        ccc = 1 - ccc
        return ccc * 10


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
