import torch
import torch.nn as nn
import torch.nn.parallel


class Shifted_softplus(nn.Module):
    """
  Performs a Shifter softplus loss, which modifies with a value of log(2)
  """

    def __init__(self):
        super(Shifted_softplus, self).__init__()
        self.act = nn.Softplus()
        self.shift = nn.Parameter(torch.tensor([0.6931]), False)

    def forward(self, X):
        """
    Applies the Activation function

    Parameters
    ----------
    node_feats: torch.Tensor
        The node features.

    Returns
    -------
    node_feats: torch.Tensor
        The updated node features.

    """
        node_feats = self.act(X) - self.shift
        return node_feats


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
