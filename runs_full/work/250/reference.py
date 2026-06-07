import torch
import torch.nn as nn


class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=20.0, eps=1e-10):
        """L2 normalization layer.

        Args:
            n_dims (int): Number of dimensions to be normalized
            scale (float, optional): Defaults to 20..
            eps (float, optional): Used to avoid division by zero.
                Defaults to 1e-10.
        """
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        """Forward function."""
        x_float = x.float()
        norm = x_float.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return (self.weight[None, :, None, None].float().expand_as(x_float) *
            x_float / norm).type_as(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_dims': 4}]
