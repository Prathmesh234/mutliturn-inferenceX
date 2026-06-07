import torch
import torch.nn as nn


def rot180(input: 'torch.Tensor') ->torch.Tensor:
    return torch.flip(input, [-2, -1])


class Rot180(nn.Module):
    """Rotate a tensor image or a batch of tensor images
    180 degrees. Input must be a tensor of shape (C, H, W)
    or a batch of tensors :math:`(*, C, H, W)`.

    Args:
        input (torch.Tensor): input tensor

    Examples:
        >>> rot180 = Rot180()
        >>> input = torch.tensor([[[
        ...    [0., 0., 0.],
        ...    [0., 0., 0.],
        ...    [0., 1., 1.]
        ... ]]])
        >>> rot180(input)
        tensor([[[[1., 1., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.]]]])
    """

    def __init__(self) ->None:
        super(Rot180, self).__init__()

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return rot180(input)

    def __repr__(self):
        return self.__class__.__name__


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
