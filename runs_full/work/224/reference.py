import torch
import torch.nn as nn
import torch.nn.parallel


class Custom_dropout(nn.Module):
    """
  An implementation for few , Given a task perform a rowise sum of 2-d
  matrix , you get a zero out the contribution of few of rows in the matrix
  Given, X a 2-d matrix consisting of row vectors (1-d) x1 , x2 ,..xn.
  Sum = x1 + 0.x2 + .. + 0.xi + .. +xn
  """

    def __init__(self, dp_rate: 'float', n_permutation: 'int'):
        """
    Parameters
    ----------
    dp_rate: float
        p value of dropout.
    """
        super(Custom_dropout, self).__init__()
        self.dropout = nn.Dropout(p=dp_rate)
        self.ones = nn.Parameter(torch.ones(n_permutation), requires_grad=False
            )

    def forward(self, layer):
        """
    Returns
    -------
    node_feats: torch.Tensor
        Updated tensor.
    """
        mask = self.dropout(self.ones).view(layer.shape[0], 1).repeat(1,
            layer.shape[1])
        return mask * layer


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dp_rate': 0.5, 'n_permutation': 4}]
