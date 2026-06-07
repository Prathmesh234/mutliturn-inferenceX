import torch
from torch import nn as nn


class Aggregate(nn.Module):
    """Pooling layer based on sum or average with optional masking.

    Args:
        axis (int): axis along which pooling is done.
        mean (bool, optional): if True, use average instead for sum pooling.
        keepdim (bool, optional): whether the output tensor has dim retained or not.

    """

    def __init__(self, axis, mean=False, keepdim=True):
        super(Aggregate, self).__init__()
        self.average = mean
        self.axis = axis
        self.keepdim = keepdim

    def forward(self, input, mask=None):
        """Compute layer output.

        Args:
            input (torch.Tensor): input data.
            mask (torch.Tensor, optional): mask to be applied; e.g. neighbors mask.

        Returns:
            torch.Tensor: layer output.

        """
        if mask is not None:
            input = input * mask[..., None]
        y = torch.sum(input, self.axis)
        if self.average:
            if mask is not None:
                N = torch.sum(mask, self.axis, keepdim=self.keepdim)
                N = torch.max(N, other=torch.ones_like(N))
            else:
                N = input.size(self.axis)
            y = y / N
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'axis': 4}]
