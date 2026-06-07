import torch
import torch.nn as nn


class Focal_loss(nn.Module):
    """
    Pytorch implementation from https://github.com/richardaecn/class-balanced-loss
    Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
        labels: A float32 tensor of size [batch, num_classes].
        logits: A float32 tensor of size [batch, num_classes].
        alpha: A float32 tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
        gamma: A float32 scalar modulating loss from hard and easy examples.
    Returns:
        focal_loss: A float32 scalar representing normalized total loss.
    """

    def __init__(self, gamma=0):
        super().__init__()
        self.cross_entropy = nn.BCEWithLogitsLoss(reduction='none')
        self.gamma = gamma

    def forward(self, logits, labels, pos_weight=1, neg_weight=1):
        ce = self.cross_entropy(logits, labels)
        alpha = labels * pos_weight + (1 - labels) * neg_weight
        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * labels * logits - self.
                gamma * torch.log1p(torch.exp(-1.0 * logits)))
        loss = modulator * ce
        weighted_loss = alpha * loss
        focal_loss = torch.mean(weighted_loss)
        return focal_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
