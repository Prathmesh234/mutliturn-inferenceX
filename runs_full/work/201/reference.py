import torch
import torch.nn as nn


def centercrop(image, w, h):
    _nt, _ct, ht, wt = image.size()
    padw, padh = (wt - w) // 2, (ht - h) // 2
    if padw > 0 and padh > 0:
        image = image[:, :, padh:-padh, padw:-padw]
    return image


class WeightedBDiceLoss(nn.Module):

    def __init__(self):
        super(WeightedBDiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_pred, y_true, weight):
        _n, _ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        weight = centercrop(weight, w, h)
        y_pred = self.sigmoid(y_pred)
        smooth = 1.0
        w, m1, m2 = weight, y_true, y_pred
        score = (2.0 * torch.sum(w * m1 * m2) + smooth) / (torch.sum(w * m1
            ) + torch.sum(w * m2) + smooth)
        loss = 1.0 - torch.sum(score)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
