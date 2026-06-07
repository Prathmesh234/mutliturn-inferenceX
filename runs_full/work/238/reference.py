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


class Atom_Wise_Convolution(nn.Module):
    """
  Performs self convolution to each node
  """

    def __init__(self, input_feature: 'int', output_feature: 'int', dropout:
        'float'=0.2, UseBN: 'bool'=True):
        """
    Parameters
    ----------
    input_feature: int
        Size of input feature size
    output_feature: int
        Size of output feature size
    dropout: float, defult 0.2
        p value for dropout between 0.0 to 1.0
    UseBN: bool
        Setting it to True will perform Batch Normalisation
    """
        super(Atom_Wise_Convolution, self).__init__()
        self.conv_weights = nn.Linear(input_feature, output_feature)
        self.batch_norm = nn.LayerNorm(output_feature)
        self.UseBN = UseBN
        self.activation = Shifted_softplus()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, node_feats):
        """
    Update node representations.

    Parameters
    ----------
    node_feats: torch.Tensor
        The node features. The shape is `(N, Node_feature_size)`.

    Returns
    -------
    node_feats: torch.Tensor
        The updated node features. The shape is `(N, Node_feature_size)`.

    """
        node_feats = self.conv_weights(node_feats)
        if self.UseBN:
            node_feats = self.batch_norm(node_feats)
        node_feats = self.activation(node_feats)
        node_feats = self.dropout(node_feats)
        return node_feats


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_feature': 4, 'output_feature': 4}]
