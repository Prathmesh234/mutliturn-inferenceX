import torch
import torch.nn as nn


class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.bce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, y_pred, y_true, weight):
        y_true_hot = y_true.argmax(1)
        loss = self.bce(y_pred, y_true_hot.long()) * weight
        return (loss.mean() * 10).clamp(0, 20)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
