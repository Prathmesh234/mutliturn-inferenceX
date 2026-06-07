import torch
from torch import nn as nn


class BehlerAngular(nn.Module):
    """
    Compute Behler type angular contribution of the angle spanned by three atoms:

    :math:`2^{(1-\\zeta)} (1 + \\lambda \\cos( {\\theta}_{ijk} ) )^\\zeta`

    Sets of zetas with lambdas of -1 and +1 are generated automatically.

    Args:
        zetas (set of int): Set of exponents used to compute angular Behler term (default={1})

    """

    def __init__(self, zetas={1}):
        super(BehlerAngular, self).__init__()
        self.zetas = zetas

    def forward(self, cos_theta):
        """
        Args:
            cos_theta (torch.Tensor): Cosines between all pairs of neighbors of the central atom.

        Returns:
            torch.Tensor: Tensor containing values of the angular filters.
        """
        angular_pos = [(2 ** (1 - zeta) * ((1.0 - cos_theta) ** zeta).
            unsqueeze(-1)) for zeta in self.zetas]
        angular_neg = [(2 ** (1 - zeta) * ((1.0 + cos_theta) ** zeta).
            unsqueeze(-1)) for zeta in self.zetas]
        angular_all = angular_pos + angular_neg
        return torch.cat(angular_all, -1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
