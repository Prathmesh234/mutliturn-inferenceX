import torch
import torch.jit
import torch.nn.functional as F
import torch.nn.functional
from functools import partial
from torch.nn.modules.loss import _Loss


def reduced_focal_loss(outputs: 'torch.Tensor', targets: 'torch.Tensor',
    threshold: 'float'=0.5, gamma: 'float'=2.0, reduction='mean'):
    """
    Compute reduced focal loss between target and output logits.
    Source https://github.com/BloodAxe/pytorch-toolbelt
    See :class:`~pytorch_toolbelt.losses` for details.
    Args:
        outputs: Tensor of arbitrary shape
        targets: Tensor of the same shape as input
        reduction (string, optional):
            Specifies the reduction to apply to the output:
            "none" | "mean" | "sum" | "batchwise_mean".
            "none": no reduction will be applied,
            "mean": the sum of the output will be divided by the number of
            elements in the output,
            "sum": the output will be summed.
            Note: :attr:`size_average` and :attr:`reduce`
            are in the process of being deprecated,
            and in the meantime, specifying either of those two args
            will override :attr:`reduction`.
            "batchwise_mean" computes mean loss per sample in batch.
            Default: "mean"
    See https://arxiv.org/abs/1903.01347
    """
    targets = targets.type(outputs.type())
    logpt = -F.binary_cross_entropy_with_logits(outputs, targets, reduction
        ='none')
    pt = torch.exp(logpt)
    focal_reduction = ((1.0 - pt) / threshold).pow(gamma)
    focal_reduction[pt < threshold] = 1
    loss = -focal_reduction * logpt
    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum()
    if reduction == 'batchwise_mean':
        loss = loss.sum(0)
    return loss


def sigmoid_focal_loss(outputs: 'torch.Tensor', targets: 'torch.Tensor',
    gamma: 'float'=2.0, alpha: 'float'=0.25, reduction: 'str'='mean'):
    """
    Compute binary focal loss between target and output logits.
    Source https://github.com/BloodAxe/pytorch-toolbelt
    See :class:`~pytorch_toolbelt.losses` for details.
    Args:
        outputs: Tensor of arbitrary shape
        targets: Tensor of the same shape as input
        reduction (string, optional):
            Specifies the reduction to apply to the output:
            "none" | "mean" | "sum" | "batchwise_mean".
            "none": no reduction will be applied,
            "mean": the sum of the output will be divided by the number of
            elements in the output,
            "sum": the output will be summed.
    See https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py  # noqa: E501
    """
    targets = targets.type(outputs.type())
    logpt = -F.binary_cross_entropy_with_logits(outputs, targets, reduction
        ='none')
    pt = torch.exp(logpt)
    loss = -(1 - pt).pow(gamma) * logpt
    if alpha is not None:
        loss = loss * (alpha * targets + (1 - alpha) * (1 - targets))
    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum()
    if reduction == 'batchwise_mean':
        loss = loss.sum(0)
    return loss


class FocalLossBinary(_Loss):

    def __init__(self, ignore: 'int'=None, reduced: 'bool'=False, gamma:
        'float'=2.0, alpha: 'float'=0.25, threshold: 'float'=0.5, reduction:
        'str'='mean'):
        """
        Compute focal loss for binary classification problem.
        """
        super().__init__()
        self.ignore = ignore
        if reduced:
            self.loss_fn = partial(reduced_focal_loss, gamma=gamma,
                threshold=threshold, reduction=reduction)
        else:
            self.loss_fn = partial(sigmoid_focal_loss, gamma=gamma, alpha=
                alpha, reduction=reduction)

    def forward(self, logits, targets):
        """
        Args:
            logits: [bs; ...]
            targets: [bs; ...]
        """
        targets = targets.view(-1)
        logits = logits.view(-1)
        if self.ignore is not None:
            not_ignored = targets != self.ignore
            logits = logits[not_ignored]
            targets = targets[not_ignored]
        loss = self.loss_fn(logits, targets)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
