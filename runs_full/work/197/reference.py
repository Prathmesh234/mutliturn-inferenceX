import torch
import torch.nn as nn


def centercrop(image, w, h):
    _nt, _ct, ht, wt = image.size()
    padw, padh = (wt - w) // 2, (ht - h) // 2
    if padw > 0 and padh > 0:
        image = image[:, :, padh:-padh, padw:-padw]
    return image


def flatten(x):
    x_flat = x.clone()
    x_flat = x_flat.view(x.shape[0], -1)
    return x_flat


class BDiceLoss(nn.Module):

    def __init__(self):
        super(BDiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_pred, y_true, weight=None):
        _n, _ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        y_pred = self.sigmoid(y_pred)
        smooth = 1.0
        y_true_f = flatten(y_true)
        y_pred_f = flatten(y_pred)
        score = (2.0 * torch.sum(y_true_f * y_pred_f) + smooth) / (torch.
            sum(y_true_f) + torch.sum(y_pred_f) + smooth)
        return 1.0 - score


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
