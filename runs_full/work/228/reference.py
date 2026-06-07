import math
import torch
import torch.nn as nn
import torch.nn.parallel


class ScaleNorm(nn.Module):
    """Apply Scale Normalization to input.

  The ScaleNorm layer first computes the square root of the scale, then computes the matrix/vector norm of the input tensor.
  The norm value is calculated as `sqrt(scale) / matrix norm`.
  Finally, the result is returned as `input_tensor * norm value`.

  This layer can be used instead of LayerNorm when a scaled version of the norm is required.
  Instead of performing the scaling operation (`scale / norm`) in a lambda-like layer, we are defining it within this layer to make prototyping more efficient.

  References
  ----------
  .. [1] Lukasz Maziarka et al. "Molecule Attention Transformer" Graph Representation Learning workshop and Machine Learning and the Physical Sciences workshop at NeurIPS 2019. 2020. https://arxiv.org/abs/2002.08264

  Examples
  --------
  >>> from deepchem.models.torch_models.layers import ScaleNorm
  >>> scale = 0.35
  >>> layer = ScaleNorm(scale)
  >>> input_tensor = torch.tensor([[1.269, 39.36], [0.00918, -9.12]])
  >>> output_tensor = layer(input_tensor)
  """

    def __init__(self, scale: 'float', eps: 'float'=1e-05):
        """Initialize a ScaleNorm layer.

    Parameters
    ----------
    scale: float
      Scale magnitude.
    eps: float
      Epsilon value. Default = 1e-5.
    """
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(math.sqrt(scale)))
        self.eps = eps

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=
            self.eps)
        return x * norm


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'scale': 1.0}]
