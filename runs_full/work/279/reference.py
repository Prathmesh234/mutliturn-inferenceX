import torch
from torch import Tensor
import torch.nn as nn


class NCutLossOptimized(nn.Module):
    """Implementation of the continuous N-Cut loss, as in:
    'W-Net: A Deep Model for Fully Unsupervised Image Segmentation', by Xia, Kulis (2017)"""

    def __init__(self, radius: 'int'=5):
        """
        :param radius: Radius of the spatial interaction term
        """
        super(NCutLossOptimized, self).__init__()
        self.radius = radius

    def forward(self, labels: 'Tensor', weights: 'Tensor') ->Tensor:
        """Computes the continuous N-Cut loss, given a set of class probabilities (labels) and image weights (weights).
        :param weights: Image pixel weights
        :param labels: Predicted class probabilities
        :return: Continuous N-Cut loss
        """
        num_classes = labels.shape[1]
        losses = []
        region_size = 2 * [2 * self.radius + 1]
        unfold = torch.nn.Unfold(region_size, padding=self.radius)
        unflatten = torch.nn.Unflatten(1, region_size)
        for k in range(num_classes):
            class_probs = labels[:, k].unsqueeze(1)
            p_f = class_probs.flatten(start_dim=1)
            P = unflatten(unfold(class_probs)).permute(0, 3, 1, 2)
            L = torch.einsum('ij,ij->i', p_f, torch.sum(weights * P, dim=(2,
                3))) / torch.einsum('ij,ij->i', p_f, torch.sum(weights, dim
                =(2, 3)))
            losses.append(nn.L1Loss()(L, torch.zeros_like(L)))
        return num_classes - sum(losses)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 16, 11, 11])]


def get_init_inputs():
    return [[], {}]
