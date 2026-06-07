import torch
import torch.nn as nn


class PixWiseBCELoss(nn.Module):

    def __init__(self, beta=0.5):
        super().__init__()
        self.criterion = nn.BCELoss()
        self.beta = beta

    def forward(self, net_mask, net_label, target_mask, target_label):
        pixel_loss = self.criterion(net_mask, target_mask)
        binary_loss = self.criterion(net_label, target_label)
        loss = pixel_loss * self.beta + binary_loss * (1 - self.beta)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
