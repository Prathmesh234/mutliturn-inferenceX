import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=0.99):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(
            euclidean_distance, 2) + label * torch.pow(torch.clamp(self.
            margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
