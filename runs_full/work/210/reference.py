import torch
import torch.nn as nn
import torch.nn.functional as F


def centercrop(image, w, h):
    _nt, _ct, ht, wt = image.size()
    padw, padh = (wt - w) // 2, (ht - h) // 2
    if padw > 0 and padh > 0:
        image = image[:, :, padh:-padh, padw:-padw]
    return image


class WeightedMCEloss(nn.Module):

    def __init__(self):
        super(WeightedMCEloss, self).__init__()

    def forward(self, y_pred, y_true, weight):
        _n, _ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        weight = centercrop(weight, w, h)
        y_pred_log = F.log_softmax(y_pred, dim=1)
        logpy = torch.sum(weight * y_pred_log * y_true, dim=1)
        loss = -torch.mean(logpy)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
